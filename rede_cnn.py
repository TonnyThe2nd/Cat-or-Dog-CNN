from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os
import cv2

path_cachorro = './Diretorio_imagens/Dog'
path_gato = './Diretorio_imagens/Cat'
def remover_imagens_invalidas(path):
    for imagem in os.listdir(path):
        caminho = os.path.join(path, imagem)
        if os.path.isfile(caminho):
            try:
                image = cv2.imread(caminho)
                if image is None:
                    print(f'Imagem: {caminho} não está correta e será excluída!')
                    os.remove(caminho)
                else:
                    print(f'Imagem: {caminho} lida com sucesso!')
            except Exception as e:
                print(f'Erro ao processar a imagem {caminho}: {e}')

remover_imagens_invalidas(path_cachorro)
remover_imagens_invalidas(path_gato)

diretorio_imagens = './Diretorio_imagens'

train_datagen = ImageDataGenerator(rescale=1./255,
                                   validation_split=0.3,
                                    rotation_range=20,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
    directory = diretorio_imagens,
    target_size=(128,128),
    batch_size=32,
    class_mode='binary',
    subset='training'
)
validation_generator = train_datagen.flow_from_directory(
    directory=diretorio_imagens,
    target_size=(128,128),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

print("Treinos e testes separados!")
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, MaxPooling2D

modelo = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Conv2D(128, 3, activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("Modelo criado e compilado!")

early_stop = EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True)
modelo_check = ModelCheckpoint('melhor_modelo.keras', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
modelo.summary()
history = modelo.fit(train_generator, epochs=30,validation_data=validation_generator, verbose=1, callbacks=[early_stop, reduce_lr,modelo_check])

import matplotlib.pyplot as plt

history_dic = history.history
loss = history_dic['loss']
val_loss = history_dic['val_loss']
accuracy = history_dic['accuracy']
val_accuracy = history_dic['val_accuracy']

print("Modelo treinado!")
from tensorflow.keras.models import load_model
modelo.save("rede_neural_tensor_04.h5")
#modelo = load_model('rede_neural_tensor.h5')
previsao = modelo.predict(validation_generator)
print(previsao)
loss, accuracy = modelo.evaluate(validation_generator)
print(f'Perda no Teste: {loss:.4f}')
print(f'Acurácia no Teste: {accuracy:.4f}')

