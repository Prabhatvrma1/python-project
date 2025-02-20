# Text Generation Using LSTM

## Overview
This project uses a Long Short-Term Memory (LSTM) neural network to generate text based on the works of William Shakespeare. The model is trained on a subset of Shakespeare's text and generates new sequences of text based on given input.

## Requirements
To run this project, you need the following dependencies:
- Python 3.x
- Pandas
- NumPy
- TensorFlow
- Random

Install the required packages using:
```bash
pip install pandas numpy tensorflow
```

## Dataset
The dataset is automatically downloaded from TensorFlow's storage:
```
https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt
```
The text is converted to lowercase and a subset from index `300000` to `800000` is used for training and generation.

## Model Training
The model is built using:
- An LSTM layer with 128 units
- A Dense output layer
- A Softmax activation function
- Categorical cross-entropy loss
- RMSprop optimizer

A sequence length of `40` characters is used with a step size of `3` to generate training data. The model is trained for `4` epochs using a batch size of `256`.

## Generating Text
The trained model is loaded using:
```python
model = tf.keras.models.load_model('textgen.keras')
```
A function `generate_txt` is used to generate text based on a given temperature. Temperature controls randomness in character selection:
- Low temperature (e.g., `0.2`) results in more predictable text.
- High temperature (e.g., `1.0`) produces more diverse text.

Example usage:
```python
print(generate_txt(300, 0.5))
```
This generates 300 characters of text based on the trained model.

## Sample Execution
The script prints generated text at different temperatures:
```python
print('------------------0.2----------')
print(generate_txt(300, 0.2))
print('------------------0.4----------')
print(generate_txt(300, 0.4))
print('------------------0.6----------')
print(generate_txt(300, 0.6))
print('------------------0.8----------')
print(generate_txt(300, 0.8))
print('------------------1.0----------')
print(generate_txt(300, 1.0))
```

## Notes
- The model must be trained before running the text generation script.
- Ensure `textgen.keras` exists before loading the model.
- Adjust the temperature to balance creativity and coherence in text generation.

## License
This project is open-source and free to use for educational purposes.

