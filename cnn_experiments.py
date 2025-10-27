import tensorflow as tf
# GPU memory management
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import os
from datetime import datetime


######### Dataset ##########################################################################################

# Load dataset
(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Split train/validation
X_valid, X_train = X_train_full[:6000] / 255.0, X_train_full[6000:] / 255.0
y_valid, y_train = y_train_full[:6000], y_train_full[6000:]

# Normalization of data for network stability
X_test = X_test / 255.0

# Reshape data to add channel dimension
X_train = X_train[..., None]   # (N, 28, 28, 1)
X_valid = X_valid[..., None]
X_test  = X_test[...,  None]

######### Model Experimentation #############################################################################

# Create all combinations of parameters

activation_list = ['relu', keras.layers.LeakyReLU(negative_slope=0.1), 'tanh']
optimizer_list = ['adam', 'sgd', 'rmsprop']
dropout_rate_list = [None, 0.3, 0.5]
l1_reg_list = [None, 0.001, 0.005]
l2_reg_list = [None, 0.001, 0.005]


param_combinations = list(itertools.product(
    activation_list,
    optimizer_list,
    dropout_rate_list,
    l1_reg_list,
    l2_reg_list
))

# Subtesting combinations of parameters to avoid crashin due to GPU memory limits
start = 0
end = 50
param_combinations = param_combinations[start:end]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create a dedicated folder for this batch’s plots
os.makedirs("results_CNN", exist_ok=True)
plot_dir = f"results_CNN/plots_{start}-{end}_{timestamp}"
os.makedirs(plot_dir, exist_ok=True)

csv_path = f"results_CNN/cnn_experiments_{start}-{end}_{timestamp}.csv"

results = []  # to store results for each experiment
run_id = 0    # counter for labeling runs


# Function to build CNN model with given parameters

def build_cnn(
    input_shape=(28, 28, 1),
    activation='relu',
    optimizer='adam',
    dropout_rate=None,
    l1_reg=None,
    l2_reg=None,
    initializer='he_normal'
):
    # Regularizer
    if l1_reg or l2_reg:
        reg = keras.regularizers.L1L2(l1=l1_reg or 0.0, l2=l2_reg or 0.0)
    else:
        reg = None

    model = keras.models.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation=activation, padding='same',
                            kernel_regularizer=reg, kernel_initializer=initializer,
                            input_shape=input_shape),
        keras.layers.MaxPooling2D((2, 2)),

        keras.layers.Conv2D(64, (3, 3), activation=activation, padding='same',
                            kernel_regularizer=reg, kernel_initializer=initializer),
        keras.layers.Flatten(),

        keras.layers.Dense(128, activation=activation, kernel_regularizer=reg,
                           kernel_initializer=initializer),
    ])

    if dropout_rate:
        model.add(keras.layers.Dropout(dropout_rate))

    model.add(keras.layers.Dense(10, activation='softmax'))

    # Compile
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    return model

# Looping over all different combinations

for activation, optimizer, dropout_rate, l1_reg, l2_reg in param_combinations:
    run_id += 1
    print(f"\n\nRunning model {run_id}/{len(param_combinations)}")
    print(f"Activation: {activation}, Optimizer: {optimizer}, Dropout: {dropout_rate}, L1: {l1_reg}, L2: {l2_reg}")
    
    # 1. Model creation
    model = build_cnn(
        activation=activation,
        optimizer=optimizer,
        dropout_rate=dropout_rate,
        l1_reg=l1_reg,
        l2_reg=l2_reg
    )

    # 2. Early stopping
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    # 3. Training
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=64,
        validation_data=(X_valid, y_valid),
        callbacks=[early_stop],
        verbose=0
    )

    # 4. Evaluation
    val_acc = max(history.history["val_accuracy"])
    val_loss = min(history.history["val_loss"])
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    # 5. Store results
    results.append({
        "Run": run_id,
        "Activation": activation.__class__.__name__ if not isinstance(activation, str) else activation,
        "Optimizer": optimizer,
        "Dropout": dropout_rate,
        "L1_reg": l1_reg,
        "L2_reg": l2_reg,
        "Val_Accuracy": val_acc,
        "Val_Loss": val_loss,
        "Test_Accuracy": test_acc,
        "Test_Loss": test_loss
    })

    # 6. Plot learning curves
    plt.figure(figsize=(7, 4))
    plt.plot(history.history["accuracy"], label="train acc")
    plt.plot(history.history["val_accuracy"], label="val acc")
    plt.title(f"Run {run_id} | Activation: {activation} | Opt: {optimizer}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    # Save the plot to file
    plot_filename = f"{plot_dir}/run_{run_id}_acc_plot.png"
    plt.savefig(plot_filename, dpi=150, bbox_inches="tight")
    plt.close()   # close the figure to free memory (important inside loops)
    print(f"Saved plot: {plot_filename}")

    # Checkpoint: save intermediate results every 10 runs
    if run_id % 10 == 0:
    df_partial = pd.DataFrame(results)
    df_partial.to_csv(csv_path, index=False)
    print(f"Checkpoint saved at run {run_id}")

    from tensorflow.keras import backend as K
    import gc
    K.clear_session()
    del model
    gc.collect()

# 7. Results summary table
df_results = pd.DataFrame(results)
df_results = df_results.sort_values(by="Val_Accuracy", ascending=False).reset_index(drop=True)

print("\n Experiment Summary (sorted by validation accuracy):")

# Save results
df_results.to_csv(csv_path, index=False)
print(f"Results saved to: {csv_path}")