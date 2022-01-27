import argparse
import numpy as np
import os
from tqdm import tqdm
from collections import defaultdict
import pickle
from code.data.data_utils import get_inv_relation
from code.data.get_paths import get_paths
from typing import *
import logging
import json
import sys
import wandb

logger = logging.getLogger()
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("[%(asctime)s \t %(message)s]",
                              "%Y-%m-%d %H:%M:%S")
ch.setFormatter(formatter)
logger.addHandler(ch)


class CBR(object):
    def __init__(self, args, train_dset, dev_dset, test_dset):
        self.args = args
        self.train_dset, self.dev_dset, self.test_dset = train_dset, dev_dset, test_dset
        self.train_qid2idx = {ex["id"]: ex_ctr for ex_ctr, ex in enumerate(self.train_dset)}
        self.dev_qid2idx = {ex["id"]: ex_ctr for ex_ctr, ex in enumerate(self.dev_dset)}
        self.test_qid2idx = {ex["id"]: ex_ctr for ex_ctr, ex in enumerate(self.test_dset)}
        self.train_paths_map = {}

    def set_paths_map(self, paths_map):
        self.train_paths_map = paths_map

    def get_programs(self, e: str, ans: str, all_paths_around_e: List[List[str]]):
        """
        Given an entity and answer, get all paths? which end at that ans node in the subgraph surrounding e
        """
        all_programs = []
        for path in all_paths_around_e:
            for l, (r, e_dash) in enumerate(path):
                if e_dash == ans:
                    # get the path till this point
                    all_programs.append([x for (x, _) in path[:l + 1]])  # we only need to keep the relations
        return all_programs

    @staticmethod
    def create_full_adj_list(adj_map, add_inverse_edges=True):
        adj_list_map = {}
        for e1_, re2_map_ in adj_map.items():
            for r_, e2_list_ in re2_map_.items():
                r_inv_ = get_inv_relation(r_)
                for e2_ in e2_list_:
                    adj_list_map.setdefault(e1_, []).append((r_, e2_))
                    if add_inverse_edges:
                        adj_list_map.setdefault(e2_, []).append((r_inv_, e1_))
        for e1_ in adj_list_map.keys():
            adj_list_map[e1_] = list(set(adj_list_map[e1_]))
        return adj_list_map

    def get_programs_from_nearest_neighbors(self, query, seed_idx, num_nn: Optional[int] = 5):
        all_programs = []
        nearest_queries = query["knn"][:num_nn]
        for knn_qid in nearest_queries:
            knn_query = self.train_dset[self.train_qid2idx[knn_qid]]
            if (knn_qid, seed_idx) not in self.train_paths_map:
                knn_adj_list = self.create_full_adj_list(knn_query["graph"]["adj_map"],
                                                         (self.args.add_inverse_edges_to_subg==1))
                paths_e = get_paths(args, knn_adj_list, knn_query["seed_entities"][seed_idx], max_len=3)
                self.train_paths_map[(knn_qid, seed_idx)] = paths_e
            else:
                paths_e = self.train_paths_map[(knn_qid, seed_idx)]
            for nn_ans in knn_query["answer"]:
                all_programs += self.get_programs(knn_query["seed_entities"][seed_idx], nn_ans, paths_e)
        return all_programs

    def rank_programs(self, list_programs: List[str]) -> List[str]:
        """
        Rank programs.
        """
        # Lets rank it simply by count:
        count_map = defaultdict(int)
        for p in list_programs:
            count_map[tuple(p)] += 1
        sorted_programs = sorted(count_map.items(), key=lambda kv: -kv[1])
        sorted_programs = [list(k) for (k, v) in sorted_programs]
        return sorted_programs

    def execute_one_program(self, query, e: str, path: List[str], depth: int, max_branch: int):
        """
        starts from an entity and executes the path by doing depth first search. If there are multiple edges with the same label, we consider
        max_branch number.
        """
        if depth == len(path):
            # reached end, return node
            return [e]
        try:
            next_entities = query["graph"]["adj_map"][e][path[depth]]
        except KeyError:
            next_entities = []
        if len(next_entities) == 0:
            # edge not present
            return []
        if len(next_entities) > max_branch:
            # select max_branch random entities
            next_entities = np.random.choice(next_entities, max_branch, replace=False).tolist()
        answers = []
        for e_next in next_entities:
            answers += self.execute_one_program(query, e_next, path, depth + 1, max_branch)
        return answers

    def execute_programs(self, query, e: str, path_list: List[List[str]], max_branch: int = 1000):
        all_answers = []
        not_executed_paths = []
        execution_fail_counter = 0
        executed_path_counter = 0
        for path in path_list:
            if executed_path_counter == self.args.max_num_programs:
                break
            ans = self.execute_one_program(query, e, path, depth=0, max_branch=max_branch)
            if ans == []:
                not_executed_paths.append(path)
                execution_fail_counter += 1
            else:
                executed_path_counter += 1
            all_answers += ans

        return all_answers, not_executed_paths, execution_fail_counter

    def rank_answers(self, list_answers: List[str]) -> List[str]:
        """
        Different ways to re-rank answers
        """
        # 1. rank based on occurrence, i.e. how many paths did end up at this entity?
        count_map = defaultdict(int)
        uniq_entities = set()
        for e in list_answers:
            count_map[e] += 1
            uniq_entities.add(e)
        sorted_entities_by_val = sorted(count_map.items(), key=lambda kv: -kv[1])
        return sorted_entities_by_val

    @staticmethod
    def get_rank_in_list(e, predicted_answers):
        rank = 0
        for i, e_to_check in enumerate(predicted_answers):
            if e == e_to_check:
                return i + 1
        return -1

    def get_hits(self, list_answers: List[str], gold_answers: List[str]) -> Tuple[float, float, float, float, float]:
        hits_1 = 0.0
        hits_3 = 0.0
        hits_5 = 0.0
        hits_10 = 0.0
        rr = 0.0
        for gold_answer in gold_answers:
            # remove all other gold answers from prediction
            filtered_answers = []
            for pred in list_answers:
                if pred in gold_answers and pred != gold_answer:
                    continue
                else:
                    filtered_answers.append(pred)

            rank = CBR.get_rank_in_list(gold_answer, filtered_answers)
            if rank > 0:
                if rank <= 10:
                    hits_10 += 1
                    if rank <= 5:
                        hits_5 += 1
                        if rank <= 3:
                            hits_3 += 1
                            if rank <= 1:
                                hits_1 += 1
                rr += 1.0 / rank
        return hits_10, hits_5, hits_3, hits_1, rr

    @staticmethod
    def get_accuracy(gold_answers: List[str], list_answers: List[str]) -> List[float]:
        all_acc = []
        for gold_ans in gold_answers:
            if gold_ans in list_answers:
                all_acc.append(1.0)
            else:
                all_acc.append(0.0)
        return all_acc

    def do_symbolic_case_based_reasoning(self, do_predict=False):
        num_programs = []
        num_answers = []
        num_non_executable_programs = []
        all_acc = []
        non_zero_ctr = 0
        hits_10, hits_5, hits_3, hits_1, mrr = 0.0, 0.0, 0.0, 0.0, 0.0
        weak_hits_10, weak_hits_5, weak_hits_3, weak_hits_1 = 0.0, 0.0, 0.0, 0.0
        per_relation_scores = {}  # map of performance per relation
        per_relation_query_count = {}
        if do_predict:
            logger.info("Running on test set")
            eval_dset = self.test_dset
        else:
            logger.info("Running on dev set")
            eval_dset = self.dev_dset

        learnt_programs = defaultdict(lambda: defaultdict(int))  # for each query relation, a map of programs to count
        # import pdb; pdb.set_trace()
        for eval_query in tqdm(eval_dset):
            non_zero_ctr_for_query, num_programs_for_query, not_executed_programs_for_query = 0, 0, 0
            answers = []
            for seed_id, se in enumerate(eval_query["seed_entities"]):
                all_programs = self.get_programs_from_nearest_neighbors(eval_query, seed_id, num_nn=self.args.k_adj)

                if all_programs is None or len(all_programs) == 0:
                    continue

                if len(all_programs) > 0:
                    non_zero_ctr_for_query += 1

                all_uniq_programs = self.rank_programs(all_programs)

                num_programs_for_query += len(all_uniq_programs)
                # Now execute the program
                answers_for_seed, _, not_executed_programs_for_seed = self.execute_programs(eval_query, se,
                                                                                            all_uniq_programs)
                answers.extend(answers_for_seed)
                not_executed_programs_for_query += not_executed_programs_for_seed

            num_programs.append(num_programs_for_query)
            num_non_executable_programs.append(not_executed_programs_for_query)
            non_zero_ctr += 1 if non_zero_ctr_for_query > 0 else 0
            answers = self.rank_answers(answers)

            r = eval_query["pattern_type"]
            if r not in per_relation_scores:
                per_relation_scores[r] = {"hits_1": 0, "hits_3": 0, "hits_5": 0, "hits_10": 0, "mrr": 0,
                                          "weak_hits_1": 0, "weak_hits_3": 0, "weak_hits_5": 0, "weak_hits_10": 0}
                per_relation_query_count[r] = 0
            per_relation_query_count[r] += 1
            if len(answers) > 0:
                acc = np.mean(self.get_accuracy(eval_query["answer"], [k[0] for k in answers]))
                _10, _5, _3, _1, rr = self.get_hits([k[0] for k in answers], eval_query["answer"])
                hits_10 += _10 / len(eval_query["answer"])
                hits_5 += _5 / len(eval_query["answer"])
                hits_3 += _3 / len(eval_query["answer"])
                hits_1 += _1 / len(eval_query["answer"])
                mrr += rr / len(eval_query["answer"])
                weak_hits_10 += 1.0 if _10 > 0 else 0.0
                weak_hits_5 += 1.0 if _5 > 0 else 0.0
                weak_hits_3 += 1.0 if _3 > 0 else 0.0
                weak_hits_1 += 1.0 if _1 > 0 else 0.0
                if args.output_per_relation_scores:
                    per_relation_scores[r]["hits_1"] += _1 / len(eval_query["answer"])
                    per_relation_scores[r]["hits_3"] += _3 / len(eval_query["answer"])
                    per_relation_scores[r]["hits_5"] += _5 / len(eval_query["answer"])
                    per_relation_scores[r]["hits_10"] += _10 / len(eval_query["answer"])
                    per_relation_scores[r]["mrr"] += rr / len(eval_query["answer"])
                    per_relation_scores[r]["weak_hits_1"] += 1.0 if _1 > 0 else 0.0
                    per_relation_scores[r]["weak_hits_3"] += 1.0 if _3 > 0 else 0.0
                    per_relation_scores[r]["weak_hits_5"] += 1.0 if _5 > 0 else 0.0
                    per_relation_scores[r]["weak_hits_10"] += 1.0 if _10 > 0 else 0.0
            else:
                acc = 0.0
            all_acc.append(acc)
            num_answers.append(len(answers))

        if args.output_per_relation_scores:
            for r, r_scores in per_relation_scores.items():
                r_scores["hits_1"] /= per_relation_query_count[r]
                r_scores["hits_3"] /= per_relation_query_count[r]
                r_scores["hits_5"] /= per_relation_query_count[r]
                r_scores["hits_10"] /= per_relation_query_count[r]
                r_scores["mrr"] /= per_relation_query_count[r]
                r_scores["weak_hits_1"] /= per_relation_query_count[r]
                r_scores["weak_hits_3"] /= per_relation_query_count[r]
                r_scores["weak_hits_5"] /= per_relation_query_count[r]
                r_scores["weak_hits_10"] /= per_relation_query_count[r]
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            out_file_name = os.path.join(args.output_dir, "per_relation_scores.json")
            with open(out_file_name, "w") as fout:
                logger.info("Writing per-relation scores to {}".format(out_file_name))
                fout.write(json.dumps(per_relation_scores, sort_keys=True, indent=4))
            out_file_name = os.path.join(args.output_dir, "per_relation_query_count.json")
            with open(out_file_name, "w") as fout:
                logger.info("Writing per-relation query counts to {}".format(out_file_name))
                fout.write(json.dumps(per_relation_query_count, sort_keys=True, indent=4))

        total_examples = len(eval_dset)
        logger.info(
            "Out of {} queries, atleast one program was returned for {} queries".format(total_examples, non_zero_ctr))
        logger.info("Avg number of programs {:3.2f}".format(np.mean(num_programs)))
        logger.info("Avg number of answers after executing the programs: {}".format(np.mean(num_answers)))
        logger.info("Accuracy (Loose): {}".format(np.mean(all_acc)))
        logger.info("Weak Hits@1 {}".format(weak_hits_1 / total_examples))
        logger.info("Weak Hits@3 {}".format(weak_hits_3 / total_examples))
        logger.info("Weak Hits@5 {}".format(weak_hits_5 / total_examples))
        logger.info("Weak Hits@10 {}".format(weak_hits_10 / total_examples))
        logger.info("Hits@1 {}".format(hits_1 / total_examples))
        logger.info("Hits@3 {}".format(hits_3 / total_examples))
        logger.info("Hits@5 {}".format(hits_5 / total_examples))
        logger.info("Hits@10 {}".format(hits_10 / total_examples))
        logger.info("MRR {}".format(mrr / total_examples))
        logger.info("Avg number of programs that do not execute per query: {:2.4f}".format(
            np.mean(num_non_executable_programs)))
        if self.args.print_paths:
            for k, v in learnt_programs.items():
                logger.info("query: {}".format(k))
                logger.info("=====" * 2)
                for rel, _ in learnt_programs[k].items():
                    logger.info((rel, learnt_programs[k][rel]))
                logger.info("=====" * 2)
        if self.args.use_wandb:
            # Log all metrics
            wandb.log({'weak_hits_1': weak_hits_1 / total_examples, 'weak_hits_3': weak_hits_3 / total_examples,
                       'weak_hits_5': weak_hits_5 / total_examples, 'weak_hits_10': weak_hits_10 / total_examples,
                       'hits_1': hits_1 / total_examples, 'hits_3': hits_3 / total_examples,
                       'hits_5': hits_5 / total_examples, 'hits_10': hits_10 / total_examples,
                       'mrr': mrr / total_examples, 'total_examples': total_examples, 'non_zero_ctr': non_zero_ctr,
                       'avg_num_prog': np.mean(num_programs), 'avg_num_ans': np.mean(num_answers),
                       'avg_num_failed_prog': np.mean(num_non_executable_programs), 'acc_loose': np.mean(all_acc)})


def load_json_dset(filenm):
    with open(filenm) as fin_:
        dset_ = json.load(fin_)
    return dset_


def main(args):
    data_dir = args.data_dir

    args.dev_file = os.path.join(data_dir, "dev.json")
    args.test_file = os.path.join(data_dir, "test.json") if not args.test_file_name \
        else os.path.join(data_dir, args.test_file_name)
    args.train_file = os.path.join(data_dir, "train.json")

    logger.info("Loading data:")
    train_dset = load_json_dset(args.train_file)
    dev_dset = load_json_dset(args.dev_file)
    test_dset = load_json_dset(args.test_file)

    logger.info("=========Config:============")
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

    symbolically_smart_agent = CBR(args, train_dset, dev_dset, test_dset)
    if args.train_paths_map is not None and os.path.exists(args.train_paths_map):
        with open(args.train_paths_map, 'rb') as fin:
            train_paths_map = pickle.load(fin)
        symbolically_smart_agent.set_paths_map(train_paths_map)
    symbolically_smart_agent.do_symbolic_case_based_reasoning(do_predict=args.test)
    if args.train_paths_map is None or os.path.exists(args.train_paths_map):
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        with open(os.path.join(args.output_dir, "train_paths_map.pkl"), 'wb') as fout:
            pickle.dump(symbolically_smart_agent.train_paths_map, fout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Collect subgraphs around entities")
    parser.add_argument("--data_dir", type=str, default="./cbr-akbc-data/")
    parser.add_argument("--train_paths_map", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./outputs/")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--test_file_name", type=str, default='')
    parser.add_argument("--max_num_programs", type=int, default=15, help="Max number of paths to consider")
    parser.add_argument("--print_paths", action="store_true")
    parser.add_argument("--k_adj", type=int, default=5,
                        help="Number of nearest neighbors to consider based on adjacency matrix")
    parser.add_argument("--use_wandb", type=int, choices=[0, 1], default=0, help="Set to 1 if using W&B")
    parser.add_argument("--output_per_relation_scores", action="store_true")
    # Path collection args
    parser.add_argument("--add_inverse_edges_to_subg", type=int, default=1)
    parser.add_argument("--num_paths_to_collect", type=int, default=1000)
    parser.add_argument("--ignore_sequential_inverse", type=int, choices=[0, 1], default=1)

    args = parser.parse_args()
    logger.info('COMMAND: %s' % ' '.join(sys.argv))
    if args.use_wandb:
        wandb.init(project='case-based-reasoning')
    main(args)
