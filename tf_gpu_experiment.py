from datasets import load_dataset
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import tensorflow as tf

from tensorflow.python.compiler.mlcompute import mlcompute
mlcompute.set_mlc_device(device_name='gpu')


def prepare_dataset(split: str):
    # processing code based on https://huggingface.co/docs/datasets/torch_tensorflow.html
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    dataset = load_dataset('glue', 'mrpc', split=split)
    dataset = dataset.map(lambda x: tokenizer(x['sentence1'], x['sentence2'], truncation=True, padding='max_length'),
                          batched=True)
    dataset = dataset.map(lambda examples: {'labels': examples['label']}, batched=True)

    dataset.set_format(type='tensorflow', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
    features = {x: dataset[x].to_tensor(shape=[None, tokenizer.model_max_length]) for x in
                ['input_ids', 'token_type_ids', 'attention_mask']}
    tfdataset = tf.data.Dataset.from_tensor_slices((features, dataset["labels"])).batch(1)
    return tfdataset


model_name = "bert-base-uncased"
model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
tfds_train_dataset = prepare_dataset('train')
tfds_test_dataset = prepare_dataset('test')

# model based on https://huggingface.co/docs/datasets/quicktour.html
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE, from_logits=True)
opt = tf.keras.optimizers.Adam(learning_rate=3e-5)
model.compile(optimizer=opt, loss=loss_fn, metrics=["accuracy"])
model.fit(tfds_train_dataset, epochs=1)
