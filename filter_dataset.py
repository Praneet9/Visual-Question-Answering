import json
import os
from collections import Counter
import yaml
import argparse

class FilterDataset():

    def __init__(self, config, data_type):
        
        self.config = config
        self.save_ans_vocab = config[data_type]['save_ans_vocab']
        self.dataset_path = config['dataset_path']
        with open(os.path.join(self.dataset_path, config[data_type]['questions']), 'r') as f:
            self.questions = json.load(f).get('questions')
        with open(os.path.join(self.dataset_path, config[data_type]['answers']), 'r') as f:
            self.answers = json.load(f).get('annotations')
        
        self.filtered_ids = {}
        self.question_txt = {}
        self.answer_txt = {}

    def get_color_n_count_ques(self):

        for ques in self.questions:
            
            lower_ques = ques['question'].lower()
            ques_id = ques['question_id']
            self.question_txt[ques_id] = lower_ques.strip()

            if (lower_ques.startswith('how many')) or \
                ('color' in lower_ques) or ('colour' in lower_ques):
                
                img_id = ques['image_id']
                self.filtered_ids[img_id] = self.filtered_ids.get(img_id, []) + [ques_id]
    
    def get_yes_no_ques(self):
        
        for item in self.answers:
            ques_id = item['question_id']
            answer = item['multiple_choice_answer']
            
            choices = [1 if (i['answer'] == answer) and \
                        (i['answer_confidence'] == 'yes') \
                        else 0 for i in item['answers']]
            
            confidence = sum(choices)/len(choices)
            
            self.answer_txt[ques_id] = {
                'text': answer,
                'confidence': confidence
            }

            if item['answer_type'] == 'yes/no':
                
                img_id = item['image_id']
                self.filtered_ids[img_id] = self.filtered_ids.get(img_id, []) + [ques_id]
    
    def filter_ques(self):

        print("Filtering yes/no questions...")
        self.get_yes_no_ques()

        print("Filtering color and count type questions...")
        self.get_color_n_count_ques()

        print("Aggregating data...")
        filtered_data = {}
        
        total_answers = 0
        skipped_answers = 0

        for img_id in self.filtered_ids.keys():
            question_ids = list(set(self.filtered_ids[img_id]))

            for question_id in question_ids:
                total_answers += 1
                ans = self.answer_txt[question_id]['text']
                if len(ans.split()) > 1:
                    skipped_answers += 1
                    continue

                filtered_data[question_id] = {
                    'image_id': img_id,
                    'question': self.question_txt[question_id].lower().replace('?', ' ?'),
                    'answer': self.answer_txt[question_id]['text'].lower(),
                    'confidence': self.answer_txt[question_id]['confidence']
                }
        print(f"{skipped_answers}/{total_answers} answers were skipped as they had more than 1 word!")
        return filtered_data
    
    def filter_ans(self, data):
        
        print("Filtering answers...")
        new_data = {}
        skip_keys = []
        for key in data.keys():
            ans = data[key]['answer'].replace('$', '').replace("'", '').replace('"', '')
            data[key]['answer'] = ans
            comma_separated = ans.split(',')
            slash_separated = ans.split('/')
            if len(comma_separated) > 2:
                skip_keys.append(key)
                for idx, text in enumerate(comma_separated):
                    new_key = f'{key}_{idx+1}'
                    new_data[new_key] = data[key].copy()
                    new_data[new_key]['answer'] = text
            elif len(slash_separated) > 2:
                skip_keys.append(key)
                for idx, text in enumerate(slash_separated):
                    new_key = f'{key}_{idx+1}'
                    new_data[new_key] = data[key].copy()
                    new_data[new_key]['answer'] = text

        for key in skip_keys:
            del data[key]

        data.update(new_data)
        
        if not self.save_ans_vocab:
            return data
        
        count_threshold = self.config['min_count_threshold']
        print(f"Filtering answers with count less than {count_threshold}")
        counts = Counter([i['answer'] for i in data.values()])
        unique_vocab = set()
        
        for key in counts.keys():
            
            cnt = counts[key]
            if cnt < count_threshold:
                continue
            
            unique_vocab.add(key)
        
        skip_keys = []
        for key in data.keys():
            
            if data[key]['answer'] not in unique_vocab:
                skip_keys.append(key)
        
        for key in skip_keys:
            del data[key]
        
        return data
    
    def save_vocab(self, data):
        
        ans_vocab = sorted(list(set([i['answer'] for i in data.values()])))
        ans_vocab = {idx:ans for idx, ans in enumerate(ans_vocab)}
        
        vocab_path = os.path.join(self.config['dataset_path'], self.config['ans_vocab_path'])
        
        with open(vocab_path, 'w') as f:
            json.dump(ans_vocab, f)
        
        print(f"Answer vocabulary saved in {vocab_path}")
        print(f"Answer vocabulary has {len(ans_vocab)} unique answers!")
    
    def filter(self):
        
        filtered_data = self.filter_ques()
        filtered_data = self.filter_ans(filtered_data)
        
        if self.save_ans_vocab:
            self.save_vocab(filtered_data)
        
        return filtered_data

def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter dataset for yes/no, color and count questions"
    )
    
    parser.add_argument(
        "--data_type", dest="data_type", type=str, required=True,
        choices=['validation_data', 'train_data'],
        help="The data to use. One of (validation_data, train_data)"
    )

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    data = FilterDataset(config, args['data_type']).filter()
