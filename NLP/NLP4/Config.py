# coding: utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import os
import time
import datetime
import json
import sys
import sklearn.metrics
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
from collections import defaultdict
import torch.nn.functional as F
from transformers import WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup, BertModel, RobertaModel
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import pickle
from REModel import REModel


MODEL_CLASSES = {
    'bert': BertModel,
    'roberta': RobertaModel,
}


IGNORE_INDEX = -100

        

class MyDataset():
    def __init__(self, prefix, data_path, h_t_limit):
        self.h_t_limit = h_t_limit

        self.data_path = data_path
        self.train_file = json.load(open(os.path.join(self.data_path, prefix+'.json')))

        self.data_train_bert_token = np.load(os.path.join(self.data_path, prefix+'_bert_token.npy'))
        # question: data_train_bert_mask代表什么？
        #data_train_bert_mask代表输入文本的attention mask，它指示哪些token是真实文本，哪些是padding部分。对于真实的token，mask值为1；对于padding的token，mask值为0。
        self.data_train_bert_mask = np.load(os.path.join(self.data_path, prefix+'_bert_mask.npy'))
        self.data_train_bert_starts_ends = np.load(os.path.join(self.data_path, prefix+'_bert_starts_ends.npy'))


    def __getitem__(self, index):
        return self.train_file[index], self.data_train_bert_token[index],   \
                self.data_train_bert_mask[index], self.data_train_bert_starts_ends[index]
 
    def __len__(self):
        return self.data_train_bert_token.shape[0]

class Accuracy(object):
    def __init__(self):
        self.correct = 0
        self.total = 0
    def add(self, is_correct):
        self.total += 1
        if is_correct:
            self.correct += 1
    def get(self):
        if self.total == 0:
            return 0.0
        else:
            return float(self.correct) / self.total
    def clear(self):
        self.correct = 0
        self.total = 0

class Config(object):
    def __init__(self, args):
        self.acc_NA = Accuracy()
        self.acc_not_NA = Accuracy()
        self.acc_total = Accuracy()

        self.args = args

        self.max_seq_length = args.max_seq_length
        self.relation_num = 97

        self.max_epoch = args.num_train_epochs

        self.evaluate_during_training_epoch = args.evaluate_during_training_epoch

        self.log_period = args.logging_steps
        # question: 这里的neg_multiple代表什么？
        #neg_multiple代表在生成训练数据时，对于每个正样本，要生成的负样本的数量。这是处理数据不平衡的一种方法，通过增加负样本的数量来让模型更好地学习区分正负样本。
        self.neg_multiple = 3
        self.warmup_ratio = 0.1

        self.data_path = args.prepro_data_dir
        self.batch_size = args.batch_size
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.lr = args.learning_rate
        # question: 为什么要设置一个h_t_limit
        #h_t_limit设置的是在一个批次数据中处理的最大实体对(head-tail对)的数量限制。这是为了控制模型输入的大小，保证即使在处理长文本时，模型的内存使用量也能维持在可控范围内。
        self.h_t_limit = 1800

        self.test_batch_size = self.batch_size * 2
        self.test_relation_limit = self.h_t_limit
        # question: 这里的self.dis2idx代表什么？
        '''self.dis2idx代表的是将实体对之间的距离(token)数量映射到一个较小的索引距离嵌入的数组。这是一种对距离进行离散化和嵌入的方法,目的是让模型能够利用实体对之间的空间关系信息。'''
        self.dis2idx = np.zeros((512), dtype='int64')
        self.dis2idx[1] = 1
        self.dis2idx[2:] = 2
        self.dis2idx[4:] = 3
        self.dis2idx[8:] = 4
        self.dis2idx[16:] = 5
        self.dis2idx[32:] = 6
        self.dis2idx[64:] = 7
        self.dis2idx[128:] = 8
        self.dis2idx[256:] = 9
        self.dis_size = 20

        self.train_prefix = args.train_prefix
        self.test_prefix = args.test_prefix

        self.checkpoint_dir = './checkpoint'
        self.fig_result_dir = './fig_result'


        if not os.path.exists("log"):
            os.mkdir("log")

        if not os.path.exists("checkpoint"):
            os.mkdir("checkpoint")
            
        if not os.path.exists("fig_result"):
            os.mkdir("fig_result")

    def load_test_data(self):
        print("Reading testing data...")
        self.rel2id = json.load(open(os.path.join(self.data_path, 'rel2id.json')))
        self.id2rel = {v: k for k,v in self.rel2id.items()}

        prefix = self.test_prefix
        print (prefix)
        self.is_test = ('test' == prefix)
        self.test_file = json.load(open(os.path.join(self.data_path, prefix+'.json')))

        self.data_test_bert_token = np.load(os.path.join(self.data_path, prefix+'_bert_token.npy'))
        self.data_test_bert_mask = np.load(os.path.join(self.data_path, prefix+'_bert_mask.npy'))
        self.data_test_bert_starts_ends = np.load(os.path.join(self.data_path, prefix+'_bert_starts_ends.npy'))


        self.test_len = self.data_test_bert_token.shape[0]
        assert(self.test_len==len(self.test_file))

        print("Finish reading")

        self.test_batches = self.data_test_bert_token.shape[0] // self.test_batch_size
        if self.data_test_bert_token.shape[0] % self.test_batch_size != 0:
            self.test_batches += 1

        self.test_order = list(range(self.test_len))
        self.test_order.sort(key=lambda x: np.sum(self.data_test_bert_token[x] > 0), reverse=True)

    def get_test_batch(self):
        context_idxs = torch.LongTensor(self.test_batch_size, self.max_seq_length).cuda()
        # question: h_mapping和t_mapping有什么区别？
        '''h_mapping和t_mapping分别表示头实体和尾实体在文本中的位置映射。这两个映射用于指示头实体和尾实体在输入序列中的位置，以便模型准确识别并处理实体关系。'''
        h_mapping = torch.Tensor(self.test_batch_size, self.test_relation_limit, self.max_seq_length).cuda()
        t_mapping = torch.Tensor(self.test_batch_size, self.test_relation_limit, self.max_seq_length).cuda()

        relation_mask = torch.Tensor(self.test_batch_size, self.h_t_limit).cuda()
        ht_pair_pos = torch.LongTensor(self.test_batch_size, self.h_t_limit).cuda()

        context_masks = torch.LongTensor(self.test_batch_size, self.max_seq_length).cuda()
   
        for b in range(self.test_batches):
            start_id = b * self.test_batch_size
            cur_bsz = min(self.test_batch_size, self.test_len - start_id)
            cur_batch = list(self.test_order[start_id : start_id + cur_bsz])

            for mapping in [h_mapping, t_mapping, relation_mask]:
                mapping.zero_()
            

            ht_pair_pos.zero_()

            # question: max_h_t_cnt有什么作用？
            '''max_h_t_cnt代表在一个批次中所有样本中头尾实体对的最大数量。它用于确定在一批数据处理过程中所需的张量大小确保所有样本的头尾实体对都能被正确处理。'''
            max_h_t_cnt = 1

            labels = []

            L_vertex = []
            titles = []
            indexes = []

            evi_nums = []
            all_test_idxs = []

            for i, index in enumerate(cur_batch):
                context_idxs[i].copy_(torch.from_numpy(self.data_test_bert_token[index, :]))
                context_masks[i].copy_(torch.from_numpy(self.data_test_bert_mask[index, :]))

                idx2label = defaultdict(list)
                ins = self.test_file[index]
                starts_pos = self.data_test_bert_starts_ends[index, :, 0]
                ends_pos = self.data_test_bert_starts_ends[index, :, 1]

                for label in ins['labels']:
                    idx2label[(label['h'], label['t'])].append(label['r'])


                L = len(ins['vertexSet'])
                titles.append(ins['title'])

                j = 0
                # question: 本代码是如何处理token数量大于512的文档的？
                '''由于BERT模型的输入限制,对于超过512个token的文档,本代码通过截断或选择性的采样策略，以确保模型输入不超过最大限制。
                这可能涉及选择文档中最关键的部分或通过某种策略平衡保留关键实体和上下文信息。'''
                test_idxs = []
                for h_idx in range(L):
                    for t_idx in range(L):
                        if h_idx != t_idx:
                            hlist = ins['vertexSet'][h_idx]
                            tlist = ins['vertexSet'][t_idx]

                            hlist = [ ( starts_pos[h['pos'][0]],  ends_pos[h['pos'][1]-1]  )  for h in hlist if ends_pos[h['pos'][1]-1]<511 ]
                            tlist = [ ( starts_pos[t['pos'][0]],  ends_pos[t['pos'][1]-1]  )  for t in tlist if ends_pos[t['pos'][1]-1]<511 ]
                            if len(hlist)==0 or len(tlist)==0:
                                continue
                            # question: 为什么要先除以len(hlist)， 再除以 (h[1] - h[0])？
                            '''这个过程是对每个实体在其包含的token上的注意力分数进行规范化处理。len(hlist)代表一个实体在文中可能对应的多个位置首先除以len(hlist)是为了在实体的不同提及间平均分配权重；
                            然后对每个实体提及再除以(h[1] - h[0])是为了在实体提及内部的token上均匀分配注意力。这样做可以确保模型不会因实体在文中的位置或长度的变化而受到影响。'''
                            for h in hlist:
                                h_mapping[i, j, h[0]:h[1]] = 1.0 / len(hlist) / (h[1] - h[0])

                            for t in tlist:
                                t_mapping[i, j, t[0]:t[1]] = 1.0 / len(tlist) / (t[1] - t[0])
                            # question: relation_mask的作用是什么？
                            '''relation_mask用于指示哪些实体对是在当前训练/评估批次中有效的。
                            它帮助模型忽略那些不应该进行关系预测的实体对，尤其是在批处理时，有些批次可能不满所需的最大实体对数量。'''
                            relation_mask[i, j] = 1

                            delta_dis = hlist[0][0] - tlist[0][0]
                            if delta_dis < 0:
                                ht_pair_pos[i, j] = -int(self.dis2idx[-delta_dis])
                            else:
                                ht_pair_pos[i, j] = int(self.dis2idx[delta_dis])

                            test_idxs.append((h_idx, t_idx))
                            j += 1


                max_h_t_cnt = max(max_h_t_cnt, j)
                label_set = {}

                for label in ins['labels']:
                    label_set[(label['h'], label['t'], label['r'])] = label['in_annotated_train']

                labels.append(label_set)

                L_vertex.append(L)
                indexes.append(index)
                all_test_idxs.append(test_idxs)


            max_c_len = self.max_seq_length

            yield {'context_idxs': context_idxs[:cur_bsz, :max_c_len].contiguous(),
                #    'context_pos': context_pos[:cur_bsz, :max_c_len].contiguous(),
                   'h_mapping': h_mapping[:cur_bsz, :max_h_t_cnt, :max_c_len],
                   't_mapping': t_mapping[:cur_bsz, :max_h_t_cnt, :max_c_len],
                   'labels': labels,
                   'L_vertex': L_vertex,
                   'titles': titles,
                   'ht_pair_pos': ht_pair_pos[:cur_bsz, :max_h_t_cnt],
                   'indexes': indexes,
                   'context_masks': context_masks[:cur_bsz, :max_c_len].contiguous(),
                   'all_test_idxs': all_test_idxs,
                   }
        
    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def get_train_batch(self, batch):
        batch_size = len(batch)
        max_length = self.max_seq_length
        h_t_limit = self.h_t_limit
        relation_num = self.relation_num
        context_idxs = torch.LongTensor(batch_size, max_length).zero_()
        h_mapping = torch.Tensor(batch_size, h_t_limit, max_length).zero_()
        t_mapping = torch.Tensor(batch_size, h_t_limit, max_length).zero_()
        relation_multi_label = torch.Tensor(batch_size, h_t_limit, relation_num).zero_()
        relation_mask = torch.Tensor(batch_size, h_t_limit).zero_()

        context_masks = torch.LongTensor(batch_size, self.max_seq_length).zero_()
        ht_pair_pos = torch.LongTensor(batch_size, h_t_limit).zero_()

        relation_label = torch.LongTensor(batch_size, h_t_limit).fill_(IGNORE_INDEX)

        for i, item in enumerate(batch):
            max_h_t_cnt = 1

            context_idxs[i].copy_(torch.from_numpy(item[1]))
            context_masks[i].copy_(torch.from_numpy(item[2]))
            starts_pos = item[3][:, 0]
            ends_pos = item[3][:, 1]

            ins = item[0]
            labels = ins['labels']
            idx2label = defaultdict(list)

            for label in labels:
                idx2label[(label['h'], label['t'])].append(label['r'])


            train_tripe = list(idx2label.keys())
            j = 0
            for (h_idx, t_idx) in train_tripe:
                if j == self.h_t_limit:
                    break
                hlist = ins['vertexSet'][h_idx]
                tlist = ins['vertexSet'][t_idx]

                hlist = [ ( starts_pos[h['pos'][0]],  ends_pos[h['pos'][1]-1]  )  for h in hlist if ends_pos[h['pos'][1]-1]<511 ]
                tlist = [ ( starts_pos[t['pos'][0]],  ends_pos[t['pos'][1]-1]  )  for t in tlist if ends_pos[t['pos'][1]-1]<511 ]
                if len(hlist)==0 or len(tlist)==0:
                    continue

                for h in hlist:
                    h_mapping[i, j, h[0]:h[1]] = 1.0 / len(hlist) / (h[1] - h[0])

                for t in tlist:
                    t_mapping[i, j, t[0]:t[1]] = 1.0 / len(tlist) / (t[1] - t[0])

                label = idx2label[(h_idx, t_idx)]

                delta_dis = hlist[0][0] - tlist[0][0]
                if delta_dis < 0:
                    ht_pair_pos[i, j] = -int(self.dis2idx[-delta_dis])
                else:
                    ht_pair_pos[i, j] = int(self.dis2idx[delta_dis])


                for r in label:
                    relation_multi_label[i, j, r] = 1

                relation_mask[i, j] = 1
                rt = np.random.randint(len(label))
                relation_label[i, j] = label[rt]

                j += 1


            lower_bound = min(len(ins['na_triple']), len(train_tripe) * self.neg_multiple)
            sel_idx = random.sample(list(range(len(ins['na_triple']))), lower_bound)
            sel_ins = [ins['na_triple'][s_i] for s_i in sel_idx]

            for (h_idx, t_idx) in sel_ins:
                if j == h_t_limit:
                    break
                hlist = ins['vertexSet'][h_idx]
                tlist = ins['vertexSet'][t_idx]

                hlist = [ ( starts_pos[h['pos'][0]],  ends_pos[h['pos'][1]-1]  )  for h in hlist if ends_pos[h['pos'][1]-1]<511 ]
                tlist = [ ( starts_pos[t['pos'][0]],  ends_pos[t['pos'][1]-1]  )  for t in tlist if ends_pos[t['pos'][1]-1]<511 ]
                if len(hlist)==0 or len(tlist)==0:
                    continue

                for h in hlist:
                    h_mapping[i, j, h[0]:h[1]] = 1.0 / len(hlist) / (h[1] - h[0])

                for t in tlist:
                    t_mapping[i, j, t[0]:t[1]] = 1.0 / len(tlist) / (t[1] - t[0])


                delta_dis = hlist[0][0] - tlist[0][0]

                relation_multi_label[i, j, 0] = 1
                relation_label[i, j] = 0
                relation_mask[i, j] = 1

                if delta_dis < 0:
                    ht_pair_pos[i, j] = -int(self.dis2idx[-delta_dis])
                else:
                    ht_pair_pos[i, j] = int(self.dis2idx[delta_dis])
                j += 1
            # question: max_h_t_cnt代表什么？
            '''max_h_t_cnt代表一个数据实例中，考虑关系预测的最大实体对(head-tail对)数量。这个数量可能受到实体数量和设置的h_t_limit限制的影响,用于在训练或测试时保持批次数据的一致性。'''
            max_h_t_cnt = max(max_h_t_cnt, len(train_tripe) + lower_bound)

        return {'context_idxs': context_idxs,
                'h_mapping': h_mapping[:, :max_h_t_cnt, :].contiguous(),
                't_mapping': t_mapping[:, :max_h_t_cnt, :].contiguous(),
                'relation_label': relation_label[:, :max_h_t_cnt].contiguous(),
                'relation_multi_label': relation_multi_label[:, :max_h_t_cnt].contiguous(),
                'relation_mask': relation_mask[:, :max_h_t_cnt].contiguous(),
                'ht_pair_pos': ht_pair_pos[:, :max_h_t_cnt].contiguous(),
                'context_masks': context_masks,
                }

    def train(self, model_type, model_name_or_path, save_name):
        self.load_test_data()

        train_dataset = MyDataset(self.train_prefix, self.data_path, self.h_t_limit)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=self.batch_size, collate_fn=self.get_train_batch, num_workers=2)

        bert_model = MODEL_CLASSES[model_type].from_pretrained(model_name_or_path)

        ori_model = REModel(config = self, bert_model=bert_model)
        ori_model.cuda()

        model = nn.DataParallel(ori_model)

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr, eps=self.args.adam_epsilon)
        tot_step = int( (len(train_dataset) // self.batch_size+1) / self.gradient_accumulation_steps * self.max_epoch)

        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(self.warmup_ratio*tot_step), num_training_steps=tot_step)

        save_step = int( (len(train_dataset) // self.batch_size+1) / self.gradient_accumulation_steps  * self.evaluate_during_training_epoch)
        print ("tot_step:", tot_step, "save_step:", save_step, self.lr)
        # question: 这里可以使用cross-entropy loss吗？
        '''可以使用交叉熵损失(cross-entropy loss),尤其是当任务被视为多分类问题时。
        在关系抽取任务中，模型预测的是一个实体对属于各个关系类别的概率，交叉熵损失可以衡量模型预测的概率分布与真实分布之间的差异。'''
        BCE = nn.BCEWithLogitsLoss(reduction='none')

        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)


        best_all_f1 = 0.0
        best_all_auc = 0
        best_all_epoch = 0

        model.train()

        global_step = 0
        total_loss = 0
        start_time = time.time()

        def logging(s, print_=True, log_=True):
            if print_:
                print(s)
            if log_:
                with open(os.path.join(os.path.join("log", save_name)), 'a+') as f_log:
                    f_log.write(s + '\n')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim(0.3, 1.0)
        plt.xlim(0.0, 0.4)
        plt.title('Precision-Recall')
        plt.grid(True)

        step = 0
        for epoch in range(self.max_epoch):

            self.acc_NA.clear()
            self.acc_not_NA.clear()
            self.acc_total.clear()

            for batch in train_dataloader:
                data = {k: v.cuda() for k,v in batch.items()}

                context_idxs = data['context_idxs']
                h_mapping = data['h_mapping']
                t_mapping = data['t_mapping']
                relation_label = data['relation_label']
                relation_multi_label = data['relation_multi_label']
                relation_mask = data['relation_mask']

                ht_pair_pos = data['ht_pair_pos']
                context_masks = data['context_masks']


                if torch.sum(relation_mask)==0:
                    print ('zero input')
                    continue
 
                dis_h_2_t = ht_pair_pos+10
                dis_t_2_h = -ht_pair_pos+10

                predict_re = model(context_idxs, h_mapping, t_mapping, dis_h_2_t, dis_t_2_h, context_masks)

                pred_loss = BCE(predict_re, relation_multi_label)*relation_mask.unsqueeze(2)

                loss = torch.sum(pred_loss) /  (self.relation_num * torch.sum(relation_mask))
                if torch.isnan(loss):
                    pickle.dump(data, open("crash_data.pkl","wb"))
                    path = os.path.join(self.checkpoint_dir, model_name+"_crash")
                    torch.save(ori_model.state_dict(), path)


                if self.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps

                output = torch.argmax(predict_re, dim=-1)
                output = output.data.cpu().numpy()

                loss.backward()

                relation_label = relation_label.data.cpu().numpy()

                for i in range(output.shape[0]):
                    for j in range(output.shape[1]):
                        label = relation_label[i][j]
                        if label < 0:
                            break

                        if label == 0:
                            self.acc_NA.add(output[i][j] == label)
                        else:
                            self.acc_not_NA.add(output[i][j] == label)

                        self.acc_total.add(output[i][j] == label)

                total_loss += loss.item()

                if (step + 1) % self.gradient_accumulation_steps == 0:            

                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                    if global_step % self.log_period == 0 :
                        cur_loss = total_loss / self.log_period
                        elapsed = time.time() - start_time
                        # question: 这里的NA acc / not NA acc / tot acc分表代表什么？是如何计算的？
                        '''NA acc代表模型在预测“非关系”(NA)样本的准确度;not NA acc代表在预测具有特定关系的样本上的准确度;tot acc代表模型在所有样本(包括有关系和无关系的）上的总准确度。
                        这些准确度通过比较模型预测和真实标签来计算。'''
                        logging('| epoch {:2d} | step {:4d} |  ms/b {:5.2f} | train loss {:.8f} | NA acc: {:4.2f} | not NA acc: {:4.2f}  | tot acc: {:4.2f} '.format(
                            epoch, global_step, elapsed * 1000 / self.log_period, cur_loss, self.acc_NA.get(), self.acc_not_NA.get(), self.acc_total.get()))
                        total_loss = 0
                        start_time = time.time()


                    if global_step % save_step == 0:
                        logging('-' * 89)
                        eval_start_time = time.time()
                        model.eval()
                        all_f1, ign_f1, f1, auc, pr_x, pr_y = self.test(model, save_name)
                        model.train()
                        logging('| epoch {:3d} | time: {:5.2f}s'.format(epoch, time.time() - eval_start_time))
                        logging('-' * 89)

                    
                        if all_f1 > best_all_f1:
                            best_all_f1 = all_f1
                            best_all_epoch = epoch
                            best_all_auc = auc
                            path = os.path.join(self.checkpoint_dir, save_name)
                            torch.save(ori_model.state_dict(), path)
                            print("Storing result...")


                step += 1


        logging('-' * 89)
        eval_start_time = time.time()
        model.eval()
        all_f1, ign_f1, f1, auc, pr_x, pr_y = self.test(model, save_name)
        model.train()
        logging('| epoch {:3d} | time: {:5.2f}s'.format(epoch, time.time() - eval_start_time))
        logging('-' * 89)

        if all_f1 > best_all_f1:
            best_all_f1 = all_f1
            best_all_epoch = epoch
            path = os.path.join(self.checkpoint_dir, save_name)
            torch.save(ori_model.state_dict(), path)
            print("Storing result...")

        print("Finish training")
        print("Best epoch = %d | F1 = %f AUC = %f" % (best_all_epoch, best_all_f1, best_all_auc))

    def test(self, model, save_name, output=False, input_theta=-1):
        data_idx = 0
        eval_start_time = time.time()
        # test_result_ignore = []
        total_recall_ignore = 0

        test_result = []
        total_recall = 0
        top1_acc = have_label = 0

        predicted_as_zero = 0
        total_ins_num = 0

        def logging(s, print_=True, log_=True):
            if print_:
                print(s)
            if log_:
                with open(os.path.join(os.path.join("log", save_name)), 'a+') as f_log:
                    f_log.write(s + '\n')

        for data in self.get_test_batch():
            with torch.no_grad():
                context_idxs = data['context_idxs']
                h_mapping = data['h_mapping']
                t_mapping = data['t_mapping']
                labels = data['labels']
                L_vertex = data['L_vertex']
                ht_pair_pos = data['ht_pair_pos']
                context_masks = data['context_masks']
                all_test_idxs = data['all_test_idxs']


                titles = data['titles']
                indexes = data['indexes']

                dis_h_2_t = ht_pair_pos+10
                dis_t_2_h = -ht_pair_pos+10

                predict_re = model(context_idxs, h_mapping, t_mapping, dis_h_2_t, dis_t_2_h, context_masks)
                # question: 这里是否可以改成softmax函数？
                '''应该是不可以，本次任务允许一个实体对属于多个关系类别。
                如果关系抽取任务被设计成每个实体对只能属于一个关系类别（互斥的类别),则使用softmax函数是适当的,因为softmax能够确保输出向量的所有元素之和为1，每个元素表示属于对应类别的概率。
                然而，如果任务允许一个实体对属于多个关系类别(非互斥的类别),则使用sigmoid函数更合适,因为它允许模型独立地评估每个类别的存在概率。
                '''
                predict_re = torch.sigmoid(predict_re)

            predict_re = predict_re.data.cpu().numpy()
            for i in range(len(labels)):
                label = labels[i]
                index = indexes[i]


                total_recall += len(label)
                for l in label.values():
                    if not l:
                        total_recall_ignore += 1

                L = L_vertex[i]
                test_idxs = all_test_idxs[i]
                j = 0

                for (h_idx, t_idx) in test_idxs:            
                    r = np.argmax(predict_re[i, j])
                    predicted_as_zero += (r==0)
                    total_ins_num += 1
                    if (h_idx, t_idx, r) in label:
                        top1_acc += 1

                    flag = False

                    for r in range(1, self.relation_num):
                        intrain = False

                        if (h_idx, t_idx, r) in label:
                            flag = True
                            if label[(h_idx, t_idx, r)]==True:
                                intrain = True
                        test_result.append( ((h_idx, t_idx, r) in label, float(predict_re[i,j,r]), intrain,  titles[i], self.id2rel[r], index, h_idx, t_idx, r) )

                    if flag:
                        have_label += 1

                    j += 1

            data_idx += 1

            if data_idx % self.log_period == 0:
                print('| step {:3d} | time: {:5.2f}'.format(data_idx // self.log_period, (time.time() - eval_start_time)))
                eval_start_time = time.time()

        test_result.sort(key = lambda x: x[1], reverse=True)

        print ('total_recall', total_recall)
        print('predicted as zero', predicted_as_zero)
        print('total ins num', total_ins_num)
        print('top1_acc', top1_acc)

        pr_x = []
        pr_y = []
        correct = 0
        w = 0

        if total_recall == 0:
            total_recall = 1  # for test

        for i, item in enumerate(test_result):
            correct += item[0]
            pr_y.append(float(correct) / (i + 1))
            pr_x.append(float(correct) / total_recall)
            if item[1] > input_theta:
                w = i


        pr_x = np.asarray(pr_x, dtype='float32')
        pr_y = np.asarray(pr_y, dtype='float32')
        f1_arr = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))
        f1 = f1_arr.max()
        f1_pos = f1_arr.argmax()
        all_f1 = f1
        theta = test_result[f1_pos][1]

        if input_theta==-1:
            w = f1_pos
            input_theta = theta

        auc = sklearn.metrics.auc(x = pr_x, y = pr_y)
        if not self.is_test:
            # question: 这里的Theta / F1 / AUC分别代表什么？是如何计算的？
            '''1.Theta指的是用于决定模型预测为正类的阈值。
               2.F1指的是模型性能的衡量指标,是精确率和召回率的调和平均值。
               3.AUC代表接收者操作特征曲线(ROC)下的面积，用于评估模型在所有阈值条件下的整体表现。
               这些指标通过对模型预测结果按照得分排序,根据不同的阈值计算出一系列的精确率和召回率,进而计算出F1分数和AUC。'''
            logging('ALL  : Theta {:3.4f} | F1 {:3.4f} | AUC {:3.4f}'.format(theta, f1, auc))
        else:
            # question: 这里的input_theta / test_result F1 / AUC分别代表什么？是如何计算的？
            ''' 1.input_theta代表用于模型预测时作为分界线的阈值. 当模型预测的概率大于等于这个阈值时, 预测结果被判定为正例; 反之, 则被判定为负例.
                input_theta是根据模型在验证集上的性能指标(如F1分数或AUC)通过调整得来的. 
                2.test_result F1代表在测试集上, 给定input_theta阈值下, 模型预测结果的F1分数. F1分数是精度和召回率的调和平均值, 用于衡量模型的整体性能.
                如何计算: 首先, 根据input_theta将模型的概率输出转换成类别预测(0或1). 然后, 计算模型的精度(precision)和召回率(recall), 并由此计算F1分数: F1 = 2 * (precision * recall) / (precision + recall).
                3.AUC代表Area Under the ROC Curve(接收者操作特征曲线下的面积), 是评估模型区分两个类别能力的性能指标. AUC值越大, 表明模型的分类性能越好.
                如何计算: AUC是通过计算不同阈值条件下模型的真正率(True Positive Rate, TPR)和假正率(False Positive Rate, FPR), 绘制ROC曲线, 然后计算该曲线下的面积得到的. 通常使用sklearn.metrics.roc_auc_score函数直接计算.'''
            logging('ma_f1 {:3.4f} | input_theta {:3.4f} test_result F1 {:3.4f} | AUC {:3.4f}'.format(f1, input_theta, f1_arr[w], auc))

        if output:
            output = [{'index': x[-4], 'h_idx': x[-3], 't_idx': x[-2], 'r_idx': x[-1], 'r': x[-5], 'title': x[-6]} for x in test_result[:w+1]]
            json.dump(output, open(save_name + "_" + self.test_prefix + "_index.json", "w"))
            print ('finish output')

        pr_x = []
        pr_y = []
        correct = correct_in_train = 0
        w = 0
        for i, item in enumerate(test_result):
            correct += item[0]
            if item[0] & item[2]:
                correct_in_train += 1
            if correct_in_train==correct:
                p = 0
            else:
                p = float(correct - correct_in_train) / (i + 1 - correct_in_train)
            pr_y.append(p)
            pr_x.append(float(correct) / total_recall)
            if item[1] > input_theta:
                w = i

        pr_x = np.asarray(pr_x, dtype='float32')
        pr_y = np.asarray(pr_y, dtype='float32')
        f1_arr = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))
        ign_f1 = f1_arr.max()

        auc = sklearn.metrics.auc(x = pr_x, y = pr_y)
        # question: 这里的input_theta / test_result F1 / AUC分别代表什么？是如何计算的？
        '''input_theta是指定的或通过验证集确定的用于分类决策的阈值。
           test_result F1表示在该阈值下的F1分数,即在测试集上按照input_theta进行分类后得到的F1分数。
           AUC同上,表示在测试集上的模型性能。
           这些指标通过在测试集上应用模型预测,并根据实际的标签与预测得分(考虑了input_theta),计算得到。'''
        logging('Ignore ma_f1 {:3.4f} | input_theta {:3.4f} test_result F1 {:3.4f} | AUC {:3.4f}'.format(ign_f1, input_theta, f1_arr[w], auc))
        
        return all_f1, ign_f1, f1_arr[w], auc, pr_x, pr_y


    def testall(self, model_type, model_name_or_path, save_name, input_theta): 
        self.load_test_data()
        bert_model = MODEL_CLASSES[model_type].from_pretrained(model_name_or_path)
        model = REModel(config = self, bert_model=bert_model)

        model.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, save_name)))
        model.cuda()
        model = nn.DataParallel(model)
        model.eval()

        self.test(model, save_name, True, input_theta)
