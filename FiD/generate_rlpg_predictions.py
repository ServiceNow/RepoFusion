import numpy as np
import os
import torch
import pickle
import argparse
import random
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import nn
from tqdm import tqdm
from transformers import AutoTokenizer
from rule_model import RuleModel
import src.data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def setup_args():
  """
  Description: Takes in the command-line arguments from user
  """
  parser = argparse.ArgumentParser()

  parser.add_argument("--seed", type=int, default=9, help="seed for reproducibility")
  parser.add_argument("--data_split", type=str, default='medium_test', help="data_split")
  parser.add_argument("--model_path", type=str, default='/repo_data/repo_FID/rlpg_models/rlpg-h', help="base directory for storing the models")
  parser.add_argument("--batch_size", type=int, default=32, help="batch size for training the classifier")
  return parser.parse_args()


def get_prediction(rule_model, info, hole_ids, hole_stats):
  pred = rule_model(info)
  for i in range(len(hole_ids)):
    hid = hole_ids[i]
    hole_prediction = pred[i] 
    sorted_rule_order = torch.argsort(hole_prediction, descending=True)
    hole_stats[hid] = sorted_rule_order.cpu().tolist()
  return hole_stats

if __name__ == '__main__':
  args = setup_args()

  #Fix seeds
  np.random.seed(args.seed)
  os.environ['PYTHONHASHSEED'] = str(args.seed)
  torch.manual_seed(args.seed)
  random.seed(args.seed)

  model_path = args.model_path
  mode = model_path.split('/')[-1]
  
  os.makedirs(os.path.join('/repo_data/repo_preprocessed_data/rlpg_predictions', args.data_split), exist_ok=True)
  out_filename = os.path.join('/repo_data/repo_preprocessed_data/rlpg_predictions', args.data_split + '_' + mode)


  # Define the model
  if mode == 'rlpg-h':
    emb_model_type = 'codebert'  
    rule_model = RuleModel(emb_model_type=emb_model_type, device=device, mode=mode)

  # Define train and val dataloaders
  kwargs = {'num_workers': 8, 'pin_memory': True} if device=='cuda' else {}
  tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
  dataset = src.data.HoleContextDataset(os.path.join('/repo_data/repo_preprocessed_data/', args.data_split), \
                                        tokenizer=tokenizer, 
                                        num_of_examples=-1) 
  print("Number of holes in dataset: ", len(dataset))
  collate_fn = src.data.HoleContextCollator(tokenizer=tokenizer)
  data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, **kwargs)

  print("=> loading checkpoint '{}'".format(model_path))
  best_model_path = os.path.join(model_path, 'best_model.th')
  rule_model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu')), strict=False)
  print("=> loaded checkpoint '{}'".format(model_path))
  rule_model.to(device)

  rule_model.eval()

  with torch.no_grad():
    hole_stats = {}

    count = 0
    for batch in tqdm(data_loader):
      if count %100 == 0:
        print("Number of holes processed: ", len(hole_stats))
      count += 1
      hole_context = batch[0].to(device)
      hole_attention_mask = batch[1].to(device)
      hole_id = batch[2]
      #print(hole_context.shape, hole_attention_mask.shape)
      hole_stats = get_prediction(rule_model, \
                                      (hole_context, hole_attention_mask), \
                                      hole_id, \
                                      hole_stats)

      if len(hole_stats)%100 == 0:
        print("Number of holes in hole_stats: ", len(hole_stats))

  with open(out_filename, 'wb') as f_out:
    pickle.dump(hole_stats, f_out)