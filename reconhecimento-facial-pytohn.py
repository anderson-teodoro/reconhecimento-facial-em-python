# Importar a biblioteca OpenCV
import cv2

# Carregar as imagens de treino
imagens_treino = []
nomes_treino = []
for nome in nomes:
    imagem = cv2.imread(nome + ".jpg")
    imagens_treino.append(imagem)
    nomes_treino.append(nome)

# Criar o modelo de reconhecimento facial
modelo_facial = cv2.face.LBPHFaceRecognizer_create()
modelo_facial.train(imagens_treino, nomes_treino)

# Capturar a imagem da câmera e realizar o reconhecimento facial
captura = cv2.VideoCapture(0)
while True:
    ret, frame = captura.read()
    rosto, confianca = modelo_facial.predict(frame)
    if rosto == -1:
        continue
    nome = nomes_treino[rosto]
    cv2.rectangle(frame, (x, y), (x + largura, y + altura), (0, 255, 0), 2)
    cv2.putText(frame, nome, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow("Reconhecimento facial", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
captura.release()
cv2.destroyAllWindows()
