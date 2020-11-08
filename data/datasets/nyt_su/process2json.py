import json
import numpy as np

def r_find(a,element):
    return len(a) - a[::-1].index(element) - 1

def get_all_index(lst=None, item=''):
    return [index for (index,value) in enumerate(lst) if value == item]

def find_entity(entity_list,sen_list):
    sen = " ".join(sen_list)
    en = " ".join(entity_list)
    for i in range(1,len(entity_list)):
        t_e1 = " ".join(entity_list[0:i])
        t_e2 = " ".join(entity_list[i:len(entity_list)])
        if t_e1 in sen and t_e2 in sen:
            t_e1_begin = sen[:sen.index(t_e1)].count(" ")
            t_e1_end = t_e1_begin+len(entity_list[0:i])-1

            # judge if entity1 in entity2
            t_e2_begin_l = sen[:sen.index(t_e2)].count(" ")
            t_e2_end_l = t_e2_begin_l+len(entity_list[i:len(entity_list)]) - 1
            t_e2_begin_r = sen[:sen.rindex(t_e2)].count(" ")
            t_e2_end_r = t_e2_begin_r + len(entity_list[i:len(entity_list)]) - 1
            if t_e2_begin_l == t_e1_begin and t_e2_begin_r == t_e1_begin:
                continue
            else:
                t_e2_begin = t_e2_begin_r if t_e2_begin_r != t_e1_begin else t_e2_begin_l
                t_e2_end = t_e2_end_r if t_e2_begin_r != t_e1_begin else t_e2_end_l
            # e1_range=np.arange(t_e1_begin,t_e1_end+1).tolist()
            # e2_range = np.arange(t_e2_begin, t_e2_end + 1).tolist()
            # same = len([x for x in e1_range if x in e2_range])
            # if same > 0:
            #     continue
            return [i,t_e1_begin,t_e1_end,t_e2_begin,t_e2_end]

    return None

pos_set = ['CC', 'DT', 'EX', 'FW', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS',
           'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP',
           'WP$', 'WRB', ]

data = []
all_label = set()
error_instance = []
rel2id = dict()
rel_cnt = 2

with open('./nyt_ori.txt', 'r',encoding="unicode_escape") as f:
    for i, line in enumerate(f):
        item = dict()
        context, label = line.rsplit('\t', 1)
        if not label.startswith('/'):
            context, label = line.rsplit(' ', 1)
            if not label.startswith('/'):
                continue

        context = context.split("\t")
        all_context = [t.split(" ") for t in context]
        result = []
        sen_begin = -1;
        sen_end = -1;
        cnt = 0
        history_token = None
        dispect = False
        for t in all_context:
            for c in t:
                try:
                    bg = c.index('.xml')
                    sen_begin = cnt + 1
                except:
                    # undo
                    print("undo")
                if c in pos_set and sen_end < 0:
                    sen_end = cnt
                    if (c == 'CD' or c == 'TO' or c == 'IN') and history_token not in pos_set:
                        dispect = True
                if dispect and c not in pos_set:
                    # deal with the 'DC' ,'TO' and 'IN' in sentence
                    sen_end = -1

                result.append(c)
                history_token = c
                cnt += 1
        sen_list = result[sen_begin:sen_end]

        entity_find = result[1:sen_begin - 3]
        link = []
        e1 = [];
        e1_idx = []
        e2 = [];
        e2_idx = []
        find_list = find_entity(entity_find,sen_list)
        if find_list is None:
            # raise Exception("can't find entity")
            print("-------------- instance error!!! ---------")
            tt = dict()
            tt['sentence'] = " ".join(sen_list)
            tt['entity'] = entity_find
            error_instance.append(tt)
            continue
        else:
            gap,t_e1_begin,t_e1_end,t_e2_begin,t_e2_end = find_list
            e1 = entity_find[0:gap]
            e2 = entity_find[gap:len(entity_find)]
            e1_idx = [t_e1_begin,t_e1_end]
            e2_idx = [t_e2_begin,t_e2_end]

        entity1 = " ".join(e1)
        entity2 = " ".join(e2)
        # if t_e1_begin < t_e2_begin:
        #     t_e1_end = t_e1_end + 2
        #     t_e2_begin = t_e2_begin + 2
        #     t_e2_end = t_e2_end + 4
        #     sen_list.insert(t_e1_begin, '<e1>')
        #     sen_list.insert(t_e1_end, '</e1>')
        #     sen_list.insert(t_e2_begin, '<e2>')
        #     sen_list.insert(t_e2_end, '</e2>')
        #     e1_idx = [t_e1_begin, t_e1_end]
        #     e2_idx = [t_e2_begin, t_e2_end]
        # elif t_e1_begin > t_e2_begin:
        #     t_e2_end = t_e2_end + 2
        #     t_e1_begin = t_e1_begin + 2
        #     t_e1_end = t_e1_end + 4
        #     sen_list.insert(t_e2_begin, '<e1>')
        #     sen_list.insert(t_e2_end, '</e1>')
        #     sen_list.insert(t_e1_begin, '<e2>')
        #     sen_list.insert(t_e1_end, '</e2>')
        #     e1_idx = [t_e1_begin, t_e1_end]
        #     e2_idx = [t_e2_begin, t_e2_end]
        # else:
        #     raise Exception("entity index error")
        if t_e1_begin == t_e2_begin:
            raise Exception("entity index error")

        item['relation'] = label[:-1]
        item['sentence'] = sen_list
        head = dict()
        head['word'] = entity1
        head['e1_begin'],head['e1_end'] = e1_idx[0],e1_idx[-1]
        tail = dict()
        tail['word'] = entity2
        tail['e2_begin'], tail['e2_end'] = e2_idx[0], e2_idx[-1]
        item['head'] = head
        item['tail'] = tail

        data.append(item)
        all_label.add(label[:-1])

        if rel2id.get(item['relation'],-1) == -1:
            rel2id[item['relation']] = rel_cnt
            rel_cnt+=1

        item['relid'] = rel2id[item['relation']]

        # rel2id[item['relation']]=rel2id.get(item['relation'],0) + 1

print("all instance num:",len(data))
print("all label num:",len(list(np.unique(all_label))))
print("error_cnt:",len(error_instance))
print("information of error:")
for pp in error_instance:
    print(pp)


with open("nyt_ori.json",'w') as f:
    json.dump(data,f)

with open('../../support_files/nyt_su_rel2id.json','w') as r:
    json.dump(rel2id,r)