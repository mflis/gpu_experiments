from tensorflow.python.compiler.mlcompute import mlcompute
# mlcompute.set_mlc_device(device_name='gpu')
from datasets import load_dataset
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import tensorflow as tf

# model_name = "bert-base-uncased"
model_name = "albert-base-v2"
model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset = load_dataset('glue', 'mrpc', split='train')


# todo  try reformer model
# cut on nr of examples, truncation, batch size

def encode(examples):
    return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding='max_length')


dataset = dataset.map(encode, batched=True)
dataset = dataset.map(lambda examples: {'labels': examples['label']}, batched=True)

dataset.set_format(type='tensorflow', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
features = {x: dataset[x].to_tensor(default_value=0, shape=[None, tokenizer.model_max_length]) for x in
            ['input_ids', 'token_type_ids', 'attention_mask']}
tfdataset = tf.data.Dataset.from_tensor_slices((features, dataset["labels"])).batch(1)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE, from_logits=True)
opt = tf.keras.optimizers.Adam(learning_rate=3e-5)
model.compile(optimizer=opt, loss=loss_fn, metrics=["accuracy"])
model.fit(tfdataset, epochs=3)
