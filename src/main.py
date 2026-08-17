import predict
import os

def main():

    test_images = []
    folder = os.path.join('teste_imagem')

    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        test_images.append(path)

    #print(test_images)

    for img_path in test_images:
        result = predict.predict(img_path)
        print(f"{img_path} -> {result}")

if __name__ == "__main__":
    main()
        