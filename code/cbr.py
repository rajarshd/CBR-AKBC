import argparse
import numpy as np
import os
from tqdm import tqdm
from collections import defaultdict
import pickle
import torch
from data.data_utils import create_vocab, load_data, get_unique_entities, \
    read_graph, get_entities_group_by_relation, get_inv_relation, load_data_all_triples
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
    def __init__(self, args, train_map, eval_map, entity_vocab, rev_entity_vocab, rel_vocab, rev_rel_vocab, eval_vocab,
                 eval_rev_vocab, all_paths, rel_ent_map):
        self.args = args
        self.eval_map = eval_map
        self.train_map = train_map
        self.all_zero_ctr = []
        self.all_num_ret_nn = []
        self.entity_vocab, self.rev_entity_vocab, self.rel_vocab, self.rev_rel_vocab = entity_vocab, rev_entity_vocab, rel_vocab, rev_rel_vocab
        self.eval_vocab, self.eval_rev_vocab = eval_vocab, eval_rev_vocab
        self.all_paths = all_paths
        self.rel_ent_map = rel_ent_map
        self.num_non_executable_programs = []

    def set_nearest_neighbor_1_hop(self, nearest_neighbor_1_hop):
        self.nearest_neighbor_1_hop = nearest_neighbor_1_hop

    @staticmethod
    def calc_sim(adj_mat: Type[torch.Tensor], query_entities: Type[torch.LongTensor]) -> Type[torch.LongTensor]:
        """
        :param adj_mat: N X R
        :param query_entities: b is a batch of indices of query entities
        :return:
        """
        query_entities_vec = torch.index_select(adj_mat, dim=0, index=query_entities)
        sim = torch.matmul(query_entities_vec, torch.t(adj_mat))
        return sim

    def get_nearest_neighbor_inner_product(self, e1: str, r: str, k: Optional[int] = 5) -> List[str]:
        try:
            nearest_entities = [self.rev_entity_vocab[e] for e in
                                self.nearest_neighbor_1_hop[self.eval_vocab[e1]].tolist()]
            # remove e1 from the set of k-nearest neighbors if it is there.
            nearest_entities = [nn for nn in nearest_entities if nn != e1]
            # making sure, that the similar entities also have the query relation
            ctr = 0
            temp = []
            for nn in nearest_entities:
                if ctr == k:
                    break
                if len(self.train_map[nn, r]) > 0:
                    temp.append(nn)
                    ctr += 1
            nearest_entities = temp
        except KeyError:
            return None
        return nearest_entities

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

    def get_programs_from_nearest_neighbors(self, e1: str, r: str, nn_func: Callable, num_nn: Optional[int] = 5):
        all_programs = []
        nearest_entities = nn_func(e1, r, k=num_nn)
        if nearest_entities is None:
            self.all_num_ret_nn.append(0)
            return None
        self.all_num_ret_nn.append(len(nearest_entities))
        zero_ctr = 0
        for e in nearest_entities:
            if len(self.train_map[(e, r)]) > 0:
                paths_e = self.all_paths[e]  # get the collected 3 hop paths around e
                nn_answers = self.train_map[(e, r)]
                for nn_ans in nn_answers:
                    all_programs += self.get_programs(e, nn_ans, paths_e)
            elif len(self.train_map[(e, r)]) == 0:
                zero_ctr += 1
        self.all_zero_ctr.append(zero_ctr)
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
        sorted_programs = [k for (k, v) in sorted_programs]
        return sorted_programs

    def execute_one_program(self, e: str, path: List[str], depth: int, max_branch: int):
        """
        starts from an entity and executes the path by doing depth first search. If there are multiple edges with the same label, we consider
        max_branch number.
        """
        if depth == len(path):
            # reached end, return node
            return [e]
        next_rel = path[depth]
        next_entities = self.train_map[(e, path[depth])]
        if len(next_entities) == 0:
            # edge not present
            return []
        if len(next_entities) > max_branch:
            # select max_branch random entities
            next_entities = np.random.choice(next_entities, max_branch, replace=False).tolist()
        answers = []
        for e_next in next_entities:
            answers += self.execute_one_program(e_next, path, depth + 1, max_branch)
        return answers

    def execute_programs(self, e: str, path_list: List[List[str]], max_branch: Optional[int] = 1000) -> List[str]:

        all_answers = []
        not_executed_paths = []
        execution_fail_counter = 0
        executed_path_counter = 0
        for path in path_list:
            if executed_path_counter == self.args.max_num_programs:
                break
            ans = self.execute_one_program(e, path, depth=0, max_branch=max_branch)
            if ans == []:
                not_executed_paths.append(path)
                execution_fail_counter += 1
            else:
                executed_path_counter += 1
            all_answers += ans

        self.num_non_executable_programs.append(execution_fail_counter)
        return all_answers, not_executed_paths

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

    def get_hits(self, list_answers: List[str], gold_answers: List[str], query: Tuple[str, str]) -> Tuple[float, float, float, float]:
        hits_1 = 0.0
        hits_3 = 0.0
        hits_5 = 0.0
        hits_10 = 0.0
        rr = 0.0
        (e1, r) = query
        all_gold_answers = self.args.all_kg_map[(e1, r)]
        for gold_answer in gold_answers:
            # remove all other gold answers from prediction
            filtered_answers = []
            for pred in list_answers:
                if pred in all_gold_answers and pred != gold_answer:
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

    def do_symbolic_case_based_reasoning(self):
        num_programs = []
        num_answers = []
        all_acc = []
        non_zero_ctr = 0
        hits_10, hits_5, hits_3, hits_1, mrr = 0.0, 0.0, 0.0, 0.0, 0.0
        per_relation_scores = {}  # map of performance per relation
        per_relation_query_count = {}
        total_examples = 0
        learnt_programs = defaultdict(lambda: defaultdict(int))  # for each query relation, a map of programs to count
        for _, ((e1, r), e2_list) in enumerate(tqdm((self.eval_map.items()))):
            # if e2_list is in train list then remove them
            # Normally, this shouldnt happen at all, but this happens for Nell-995.
            orig_train_e2_list = self.train_map[(e1, r)]
            temp_train_e2_list = []
            for e2 in orig_train_e2_list:
                if e2 in e2_list:
                    continue
                temp_train_e2_list.append(e2)
            self.train_map[(e1, r)] = temp_train_e2_list
            # also remove (e2, r^-1, e1)
            r_inv = get_inv_relation(r, args.dataset_name)
            temp_map = {}  # map from (e2, r_inv) -> outgoing nodes
            for e2 in e2_list:
                temp_map[(e2, r_inv)] = self.train_map[e2, r_inv]
                temp_list = []
                for e1_dash in self.train_map[e2, r_inv]:
                    if e1_dash == e1:
                        continue
                    else:
                        temp_list.append(e1_dash)
                self.train_map[e2, r_inv] = temp_list

            total_examples += len(e2_list)
            all_programs = self.get_programs_from_nearest_neighbors(e1, r, self.get_nearest_neighbor_inner_product,
                                                                    num_nn=self.args.k_adj)

            if all_programs is None or len(all_programs) == 0:
                all_acc.append(0.0)
                continue

            # filter the program if it is equal to the query relation
            temp = []
            for p in all_programs:
                if len(p) == 1 and p[0] == r:
                    continue
                temp.append(p)
            all_programs = temp

            if len(all_programs) > 0:
                non_zero_ctr += len(e2_list)

            all_uniq_programs = self.rank_programs(all_programs)

            for u_p in all_uniq_programs:
                learnt_programs[r][u_p] += 1

            num_programs.append(len(all_uniq_programs))
            # Now execute the program
            answers, not_executed_programs = self.execute_programs(e1, all_uniq_programs)

            answers = self.rank_answers(answers)
            if len(answers) > 0:
                acc = self.get_accuracy(e2_list, [k[0] for k in answers])
                _10, _5, _3, _1, rr = self.get_hits([k[0] for k in answers], e2_list, query=(e1, r))
                hits_10 += _10
                hits_5 += _5
                hits_3 += _3
                hits_1 += _1
                mrr += rr
                if args.output_per_relation_scores:
                    if r not in per_relation_scores:
                        per_relation_scores[r] = {"hits_1": 0, "hits_3": 0, "hits_5": 0, "hits_10": 0, "mrr": 0}
                        per_relation_query_count[r] = 0
                    per_relation_scores[r]["hits_1"] += _1
                    per_relation_scores[r]["hits_3"] += _3
                    per_relation_scores[r]["hits_5"] += _5
                    per_relation_scores[r]["hits_10"] += _10
                    per_relation_scores[r]["mrr"] += rr
                    per_relation_query_count[r] += len(e2_list)
            else:
                acc = [0.0] * len(e2_list)
            all_acc += acc
            num_answers.append(len(answers))
            # put it back
            self.train_map[(e1, r)] = orig_train_e2_list
            for e2 in e2_list:
                self.train_map[(e2, r_inv)] = temp_map[(e2, r_inv)]

        if args.output_per_relation_scores:
            for r, r_scores in per_relation_scores.items():
                r_scores["hits_1"] /= per_relation_query_count[r]
                r_scores["hits_3"] /= per_relation_query_count[r]
                r_scores["hits_5"] /= per_relation_query_count[r]
                r_scores["hits_10"] /= per_relation_query_count[r]
                r_scores["mrr"] /= per_relation_query_count[r]
            out_file_name = os.path.join(args.output_dir, "per_relation_scores.json")
            fout = open(out_file_name, "w")
            logger.info("Writing per-relation scores to {}".format(out_file_name))
            fout.write(json.dumps(per_relation_scores, sort_keys=True, indent=4))
            fout.close()

        logger.info(
            "Out of {} queries, atleast one program was returned for {} queries".format(total_examples, non_zero_ctr))
        logger.info("Avg number of programs {:3.2f}".format(np.mean(num_programs)))
        logger.info("Avg number of answers after executing the programs: {}".format(np.mean(num_answers)))
        logger.info("Accuracy (Loose): {}".format(np.mean(all_acc)))
        logger.info("Hits@1 {}".format(hits_1 / total_examples))
        logger.info("Hits@3 {}".format(hits_3 / total_examples))
        logger.info("Hits@5 {}".format(hits_5 / total_examples))
        logger.info("Hits@10 {}".format(hits_10 / total_examples))
        logger.info("MRR {}".format(mrr / total_examples))
        logger.info("Avg number of nn, that do not have the query relation: {}".format(
            np.mean(self.all_zero_ctr)))
        logger.info("Avg num of returned nearest neighbors: {:2.4f}".format(np.mean(self.all_num_ret_nn)))
        logger.info("Avg number of programs that do not execute per query: {:2.4f}".format(
            np.mean(self.num_non_executable_programs)))
        if self.args.print_paths:
            for k, v in learnt_programs.items():
                logger.info("query: {}".format(k))
                logger.info("=====" * 2)
                for rel, _ in learnt_programs[k].items():
                    logger.info((rel, learnt_programs[k][rel]))
                logger.info("=====" * 2)
        if self.args.use_wandb:
            # Log all metrics
            wandb.log({'hits_1': hits_1 / total_examples, 'hits_3': hits_3 / total_examples,
                       'hits_5': hits_5 / total_examples, 'hits_10': hits_10 / total_examples,
                       'mrr': mrr / total_examples, 'total_examples': total_examples, 'non_zero_ctr': non_zero_ctr,
                       'all_zero_ctr': self.all_zero_ctr, 'avg_num_nn': np.mean(self.all_num_ret_nn),
                       'avg_num_prog': np.mean(num_programs), 'avg_num_ans': np.mean(num_answers),
                       'avg_num_failed_prog': np.mean(self.num_non_executable_programs), 'acc_loose': np.mean(all_acc)})


def main(args):
    dataset_name = args.dataset_name
    logger.info("==========={}============".format(dataset_name))
    data_dir = os.path.join(args.data_dir, "data", dataset_name)
    subgraph_dir = os.path.join(args.data_dir, "subgraphs", dataset_name)
    kg_file = os.path.join(data_dir, "graph.txt")

    args.dev_file = os.path.join(data_dir, "dev.txt")
    args.test_file = os.path.join(data_dir, "test.txt") if not args.test_file_name \
        else os.path.join(data_dir, args.test_file_name)
    if args.dataset_name == "FB122":
        args.test_file = os.path.join(data_dir, "testI.txt")

    args.train_file = os.path.join(data_dir, "train.txt")

    logger.info("Loading subgraph around entities:")
    with open(os.path.join(subgraph_dir, args.subgraph_file_name), "rb") as fin:
        all_paths = pickle.load(fin)

    entity_vocab, rev_entity_vocab, rel_vocab, rev_rel_vocab = create_vocab(kg_file)
    logger.info("Loading train map")
    train_map = load_data(kg_file)
    logger.info("Loading dev map")
    dev_map = load_data(args.dev_file)
    logger.info("Loading test map")
    test_map = load_data(args.test_file)
    eval_map = dev_map
    eval_file = args.dev_file
    if args.test:
        eval_map = test_map
        eval_file = args.test_file

    rel_ent_map = get_entities_group_by_relation(args.train_file)
    # Calculate nearest neighbors
    adj_mat = read_graph(kg_file, entity_vocab, rel_vocab)
    adj_mat = np.sqrt(adj_mat)
    l2norm = np.linalg.norm(adj_mat, axis=-1)
    l2norm[0] += np.finfo(np.float).eps  # to encounter zero values. These 2 indx are PAD / NULL
    l2norm[1] += np.finfo(np.float).eps
    adj_mat = adj_mat / l2norm.reshape(l2norm.shape[0], 1)

    # Lets put this to GPU
    adj_mat = torch.from_numpy(adj_mat)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    logger.info('Using device:'.format(device.__str__()))
    adj_mat = adj_mat.to(device)

    # get the unique entities in eval set, so that we can calculate similarity in advance.
    eval_entities = get_unique_entities(eval_file)
    eval_vocab, eval_rev_vocab = {}, {}
    query_ind = []

    e_ctr = 0
    for e in eval_entities:
        try:
            query_ind.append(entity_vocab[e])
        except KeyError:
            continue
        eval_vocab[e] = e_ctr
        eval_rev_vocab[e_ctr] = e
        e_ctr += 1

    logger.info("=========Config:============")
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

    logger.info("Loading combined train/dev/test map for filtered eval")
    all_kg_map = load_data_all_triples(args.train_file, args.dev_file, args.test_file)
    args.all_kg_map = all_kg_map

    symbolically_smart_agent = CBR(args, train_map, eval_map, entity_vocab, rev_entity_vocab, rel_vocab,
                                                 rev_rel_vocab, eval_vocab, eval_rev_vocab, all_paths, rel_ent_map)

    query_ind = torch.LongTensor(query_ind).to(device)
    # Calculate similarity
    sim = symbolically_smart_agent.calc_sim(adj_mat,
                                            query_ind)  # n X N (n== size of dev_entities, N: size of all entities)

    nearest_neighbor_1_hop = np.argsort(-sim.cpu(), axis=-1)
    symbolically_smart_agent.set_nearest_neighbor_1_hop(nearest_neighbor_1_hop)

    logger.info("Loaded...")

    symbolically_smart_agent.do_symbolic_case_based_reasoning()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Collect subgraphs around entities")
    parser.add_argument("--dataset_name", type=str, help="The dataset name. Replace with one of FB122 | WN18RR | NELL-995 to reproduce the results of the paper")
    parser.add_argument("--data_dir", type=str, default="./cbr-akbc-data/")
    parser.add_argument("--subgraph_file_name", type=str,
                        default="paths_1000.pkl")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--test_file_name", type=str, default='')
    parser.add_argument("--max_num_programs", type=int, default=15, help="Max number of paths to consider")
    parser.add_argument("--print_paths", action="store_true")
    parser.add_argument("--k_adj", type=int, default=5,
                        help="Number of nearest neighbors to consider based on adjacency matrix")
    parser.add_argument("--use_wandb", type=int, choices=[0, 1], default=0, help="Set to 1 if using W&B")
    parser.add_argument("--output_per_relation_scores", action="store_true")

    args = parser.parse_args()
    logger.info('COMMAND: %s' % ' '.join(sys.argv))
    if args.use_wandb:
        wandb.init(project='case-based-reasoning')
    main(args)