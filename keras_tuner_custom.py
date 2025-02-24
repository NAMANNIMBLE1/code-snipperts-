def build_model(hp):
    counter = 0
    for i in range(hp.Int('num_layers', 2, 20)):
        if counter == 0:
            model = Sequential()
            model.add(Input(shape = (100,) ))
            model.add(Dense(units=hp.Int('units_' + str(i), min_value=1, max_value=512, step=32), activation=hp.Choice('activation', values=['relu', 'tanh', 'sigmoid'])) )
            model.add(Dropout(0.5)) 
            counter += 1
        else:
            model.add(Dense(units=hp.Int('units_' + str(i), min_value=1, max_value=512, step=32), activation=hp.Choice('activation', values=['relu', 'tanh', 'sigmoid'])) )
            model.add(Dropout(0.5))
    model.add(Dense(2, activation='sigmoid'))

    hp_learning_rate = hp.Choice('learning_rate', values=[0.01, 0.001, 0.0001])
    hp_optimizer = hp.Choice('optimizer', values=['adam', 'rmsprop', 'sgd'])
    if hp_optimizer == 'adam':
        optimizer = Adam(learning_rate=hp_learning_rate)
    elif hp_optimizer == 'rmsprop':
        optimizer = RMSprop(learning_rate=hp_learning_rate)
    else:
        optimizer = SGD(learning_rate=hp_learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model


tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=3,
    directory='',
    project_name='')

tuner.search_space_summary()    
tuner.search(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
tuner.results_summary()
best_model = tuner.get_best_models(num_models=1)[0] # get the best model num_models=1 means only one model [0] means the first model
best_model.summary()

