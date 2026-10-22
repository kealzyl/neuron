# Kealzyl Neuron

A local Flask dashboard for a small PyTorch phrase classifier.

## What it does

- Classifies phrases into `yes`, `no`, `maybe`, `surely not`, or `of course`.
- Uses a PyTorch network with **10 hidden Linear layers, each containing 10 neurons**.
- Lets you add labeled training phrases in the browser.
- Trains the network and saves its weights and optimizer state.
- Lets you explicitly save or reload the saved checkpoint.
- Stores training examples in `training_data.json`.

This is a small feature-based demonstration classifier, not a large language model. Its predictions may be wrong, especially for phrases that do not resemble the training examples.

## Install and run (Windows)

1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Install dependencies:

   ```powershell
   py -m pip install -r requirements.txt
   ```

4. Start the app:

   ```powershell
   py kealzyl_neuron.py
   ```

5. Open **http://127.0.0.1:5000** in your browser.

To stop the server, return to the terminal and press `Ctrl+C`.

## How to use it

1. Try a phrase in **Test a phrase**.
2. Add examples with the correct response in **Train the network**.
3. Choose the number of training epochs and press **Train model**.
4. Test your phrases again.
5. Use **Save model** to save weights and optimizer state. Training also saves automatically.
6. Use **Load model** to reload the saved weights later.

## Files created while running

- `kealzyl_neuron.pth` — model weights.
- `kealzyl_neuron_optimizer.pth` — optimizer state, useful for continued training.
- `training_data.json` — your labeled examples.

The model checkpoint files are created in this project folder. Keep them if you want to preserve your trained model.

## Architecture

- 10 numeric phrase features
- 10 hidden layers × 10 neurons per layer
- ReLU activation after each hidden layer
- 5 output scores, one for each response class
- Cross-entropy loss and Adam optimizer

The features are transparent counts of selected words and punctuation. Adding good, varied examples is important; the model does not understand language like a general-purpose LLM.


## Importing many training examples at once

In the dashboard, paste one example per line in the **Bulk import** box:

```text
yes | Absolutely, let's do it!
no | I don't agree with that
maybe | It depends on the situation
surely not | There is no chance
of course | Without a doubt
```

You can also use a tab between the label and phrase. The label can be either first or last. Blank lines and lines beginning with `#` are ignored. Bulk import accepts up to 5,000 rows per request. After importing, click **Train model** to teach the network from the expanded dataset.

The training examples are stored in `training_data.json`, so they remain available after restarting the app.
