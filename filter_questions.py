import json
import os


class FilterQuestions():

    def __init__(self, dataset_path, questions_path, answers_path):

        with open(os.path.join(dataset_path, questions_path), 'r') as f:
            self.questions = json.load(f).get('questions')
        with open(os.path.join(dataset_path, answers_path), 'r') as f:
            self.answers = json.load(f).get('annotations')
        
        self.filtered_ids = {}
        self.question_txt = {}
        self.answer_txt = {}

    def get_color_n_count_ques(self):

        for ques in self.questions:
            
            lower_ques = ques['question'].lower()
            ques_id = ques['question_id']
            self.question_txt[ques_id] = lower_ques.strip()

            if (lower_ques.startswith('how')) or \
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
    
    def filter(self):

        print("Filtering yes/no questions...")
        self.get_yes_no_ques()

        print("Filtering color and count type questions...")
        self.get_color_n_count_ques()

        print("Aggregating data...")
        filtered_data = {}
        
        total_answers = 0
        skipped_answers = 0
        ans_vocab = set()

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
                    'question': self.question_txt[question_id],
                    'answer': self.answer_txt[question_id]['text'],
                    'confidence': self.answer_txt[question_id]['confidence']
                }
                ans_vocab.add(ans)
        
        print(f"{skipped_answers}/{total_answers} answers were skipped as they had more than 1 word!")
        print(f"Answer vocab len: {len(ans_vocab)}")
        return filtered_data


if __name__ == '__main__':

    dataset_path = 'dataset'
    questions_path = 'v2_OpenEnded_mscoco_val2014_questions.json'
    answers_path = 'v2_mscoco_val2014_annotations.json'

    data = FilterQuestions(dataset_path, questions_path, answers_path).filter()
