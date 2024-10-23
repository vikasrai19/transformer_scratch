import json
from data_model.cornell_dataset import cornell_dataset, conversation_data
from config import get_config

config = get_config()

count = 0
conv_list = []
for ind, conv in enumerate(conversation_data):
    if len(conv[1]) < config['seq_len']:
        conv_list.append(conv[1])
    else:
        count += 1

print("len of conversation ", len(conv_list))
print("total skipped conv ", count)

with open("./datasets/chat6.json", "w") as fl:
    json.dump(conv_list, fl)