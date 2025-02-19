from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
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
modelo.summary()
modelo.fit(train_generator, epochs=20,validation_data=validation_generator, verbose=1, callbacks=[early_stop,modelo_check])

print("Modelo treinado!")
from tensorflow.keras.models import load_model
modelo.save("rede_neural_tensor_02.h5")
#modelo = load_model('rede_neural_tensor.h5')
previsao = modelo.predict(validation_generator)
print(previsao)
loss, accuracy = modelo.evaluate(validation_generator)
print(f'Perda no Teste: {loss:.4f}')
print(f'Acurácia no Teste: {accuracy:.4f}')

########################

from tensorflow.keras.models import load_model

rede = load_model('rede_neural_tensor.h5')

from tensorflow.keras.preprocessing.image import load_img, img_to_array

imagem = load_img('download (1).jpg',target_size=(128,128))

import matplotlib.pyplot as plt

plt.imshow(imagem)

imagem_array = img_to_array(imagem)/255

print(imagem_array)
import numpy as np

imagem_ajustada = np.expand_dims(imagem_array, axis=0)

previsao = rede.predict(imagem_ajustada)
print(previsao)
if(previsao[0][0] > 0.5):
    print("A Imagem processada é um cachorro!")
else:
    print("A Imagem processada é um gato!")


