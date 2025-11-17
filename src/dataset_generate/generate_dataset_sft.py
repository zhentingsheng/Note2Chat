import json
import os
import argparse
from prompts.doctor import get_doctor_system_prompt

class ConversationProcessor:
    def __init__(self, dataset_dir,  trainset_path, conv_key='conv_revised', sampling_conv_key=None):
        self.dataset_dir = dataset_dir
        self.trainset_path = trainset_path
        self.trainset = []
        self.conv_key = conv_key
        self.sampling_conv_key = sampling_conv_key

    def format_system_turn(self):
        return {
            "role": "system",
            "content": get_doctor_system_prompt('v1')
        }

    def format_doctor_turn(self, turn):
        turn['role'] = 'assistant'
        return turn

    def format_patient_turn(self, turn):
        turn['role'] = 'user'
        return turn

    def build_sharegpt_sample(self, conversation):
        system_turn = self.format_system_turn()
        conversation_sharegpt = [system_turn]
        
        for i, turn in enumerate(conversation):
            if 'turn' in turn:
                del turn['turn']

            if turn['role'] in ['patient', 'user']:
                conversation_sharegpt.append(self.format_patient_turn(turn))
            elif turn['role'] in ['doctor', 'assistant']:
                conversation_sharegpt.append(self.format_doctor_turn(turn))

        return {"conversations": conversation_sharegpt}

    def process_dataset(self, file_paths):

        dataset = []
        for path in file_paths:
            with open(path, 'r', encoding='utf-8') as f:
                sample = json.load(f)
            
            if self.sampling_conv_key and sample.get(self.sampling_conv_key):
                conv_key = self.sampling_conv_key
            else:
                conv_key = self.conv_key
        
            conversation = sample[conv_key]['conversation']

            sample_sharegpt = self.build_sharegpt_sample(conversation)
            if sample_sharegpt:
                dataset.append(sample_sharegpt)

        return dataset

    def gather_files(self, input_dir):
        files = []
        if input_dir and os.path.exists(input_dir):
            for root, _, filenames in os.walk(input_dir):
                for file in filenames:
                    files.append(os.path.join(root, file))
        return files

    def process(self):
        train_files = self.gather_files(self.dataset_dir)

        print(f"Train samples: {len(train_files)}")

        self.trainset = self.process_dataset(train_files)

        self.save_output(self.trainset_path, self.trainset)

    def save_output(self, output_path, dataset):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--dataset_dir',type=str, required=True, help="dataset_dir")
    parser.add_argument('--trainset_path',type=str, required=True, help="trainset_path")
    parser.add_argument('--conv_key',type=str, required=True, help="conv_key")
    parser.add_argument('--sampling_conv_key',default=None, required=True, help="sampling_conv_key")

    args = parser.parse_args()
    dataset_dir = args.dataset_dir
    trainset_path = args.trainset_path
    conv_key = args.conv_key
    sampling_conv_key = args.sampling_conv_key

    trainset_dir = os.path.dirname(trainset_path)
    if not os.path.exists(trainset_dir):
        os.makedirs(trainset_dir)

    processor = ConversationProcessor(
        dataset_dir=dataset_dir,
        trainset_path=trainset_path,
        conv_key=conv_key,
        sampling_conv_key=sampling_conv_key
    )
    processor.process()
