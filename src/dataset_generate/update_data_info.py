import json
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--trainset_name',type=str, required=True, help="trainset_name")
    parser.add_argument('--filename',type=str, required=True, help="filename")
    parser.add_argument('--formatting',type=str, required=True, help="formatting")
    parser.add_argument('--data_info_path',type=str, required=True, help="data_info_path")

    args = parser.parse_args()
    
    trainset_name = args.trainset_name

    print(trainset_name)
    filename = args.filename
    formatting = args.formatting
    data_info_path = args.data_info_path

    dataset_info = {
        "file_name": filename
    }

    if formatting == 'sharegpt':
        dataset_info['formatting'] = 'sharegpt'
        dataset_info['tags'] = {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag": "system"
        }
    
    with open(data_info_path, 'r', encoding='utf-8') as f:
        data_info = json.load(f)

    data_info[trainset_name] = dataset_info

    with open(data_info_path, 'w', encoding='utf-8') as f:
        json.dump(data_info, f, indent=4)