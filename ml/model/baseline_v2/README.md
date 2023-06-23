tf.keras.Sequential([\
        tf.keras.layers.Conv2D(filters=11,
                               kernel_size=(3, 3),
                               activation='relu',
                               input_shape=(48, 48, 1)),\
        tf.keras.layers.Conv2D(filters=11,
                               kernel_size=(3, 3),
                               activation='relu'),\
        tf.keras.layers.BatchNormalization(),\
        tf.keras.layers.Conv2D(filters=11,
                               kernel_size=(3, 3),
                               activation='relu'),\
        tf.keras.layers.Conv2D(filters=11,
                               kernel_size=(3, 3),
                               activation='relu'),\
        tf.keras.layers.Conv2D(filters=11,
                               kernel_size=(3, 3),
                               activation='relu'),\
        tf.keras.layers.Flatten(),\
        tf.keras.layers.Dense(7, activation='softmax'),\
    ])

model.compile(\
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),\
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),\
        metrics=['accuracy']\
    )

EPOCHS = 30\
batch_size=100

Train acc: 0.916867196559906\
Val acc: 0.7075471878051758\
Test acc: 0.7075471878051758