from torchmetrics.text import CharErrorRate as CER
import json
from tqdm import tqdm

words_iam = json.load(open('data/synth/IAM.json', 'r'))
words_iam = list(words_iam['words'])
print(len(words_iam))

with open(f'outputs/iam/vals_pred_iam.txt', 'r') as f:
  outputs = f.readlines()

preds, labels = [], []
dict_preds = dict()
preds_by_freq = dict()
for idx, output in tqdm(enumerate(outputs)):
  # Output is in format: "Pred: 'word' - Label: 'word'"
  # Get pred and label
  output = output.split(" - ")[0:2]
  pred, label = output[0], output[1]
  pred = pred.split(":")[1][1:]
  label = label.split(":")[1][1:]
  
  # print(f'Pred: {pred} - Label: {label}')
  # See if label is a list of words, else set empty string
  if pred not in dict_preds:
    dict_preds[pred] = 1
  else:
    dict_preds[pred] += 1

  if pred not in preds_by_freq:
    preds_by_freq[pred] = [[], []] # Saving preds and labels
    
  preds_by_freq[pred][0].append(pred)
  preds_by_freq[pred][1].append(label)

cer = CER()
nearest_cer = CER()


sorted_by_freq = sorted(dict_preds.items(), key=lambda x:x[1], reverse=True)
print(f'Sorted by freq: {sorted_by_freq[:20]}')

best_words_mem = dict()

for pred, _ in tqdm(sorted_by_freq):
  print(f'Pred: {pred}')
  # print(preds_by_freq[pred])
  # print(preds_by_freq[pred][0])
  # print(preds_by_freq[pred][1])
  for output, label in zip(preds_by_freq[pred][0],preds_by_freq[pred][1]):
    cer_word = cer.forward(output, label)
    best_word, best_cer = output, cer_word
    if output in best_words_mem:
      # print(f'Found in the memoization structure {output}. {best_words_mem[output]}')
      best_word, best_cer = best_words_mem[output]
      best_cer = best_cer
    else:
      for word in sorted(words_iam):    
        cer_word_IAM = CER()(output, word)
        # print(f'Word: {word}. output: {output}. CER: {cer_word_IAM}')
        if cer_word_IAM < best_cer:
          best_word, best_cer = word, cer_word_IAM

      if output not in best_words_mem:
        best_words_mem[output] = (best_word, best_cer)

      best_word_cer = nearest_cer.forward(best_word, label)

    # print(f'The best word for output: {output} and label {label} is `{best_word}` with cer {best_word_cer}. Otherwise the cer would have been {cer_word}.')

print(f'Nearest CER: {nearest_cer.compute()}'
      f'Overall CER: {cer.compute()}')