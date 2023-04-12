import * as tf from '@tensorflow/tfjs-node'
import * as fs from 'fs'
import * as path from 'path'
import * as mobilenet from '@tensorflow-models/mobilenet'

// 데이터셋 경로
const datasetDir = '/path/to/dataset'

// 이미지 전처리 함수
function preprocessImage(image: tf.Tensor3D | tf.Tensor4D): tf.Tensor {
  // 이미지 크기 조정
  const resizedImage = tf.image.resizeBilinear(image, [224, 224])

  // 이미지 픽셀 값을 0~1로 정규화
  const normalizedImage = resizedImage.div(255)

  return normalizedImage
}

// 데이터셋 로드 및 전처리
function loadDataset() {
  // 이미지 파일 리스트 가져오기
  const imageFiles = fs
    .readdirSync(datasetDir)
    .filter((file) => path.extname(file).toLowerCase() === '.jpg')

  // 라벨 생성
  const labels = [
    '검정',
    '파랑',
    '갈색',
    '녹색',
    '회색',
    '주황',
    '분홍',
    '보라',
    '빨강',
    '흰색',
    '노랑'
  ]

  // 데이터셋 생성
  const dataset = tf.data
    .array(imageFiles)
    .map((file) => {
      // 이미지 파일 읽기
      const imageData = fs.readFileSync(path.join(datasetDir, file))

      // 이미지 텐서 생성
      const imageTensor = tf.node.decodeImage(imageData)

      // 이미지 전처리
      const preprocessedImage = preprocessImage(imageTensor)

      // 라벨 원-핫 인코딩
      const oneHot = tf.oneHot(labels.indexOf(path.parse(file).name.split('_')[0]), labels.length)

      return { xs: preprocessedImage, ys: oneHot }
    })
    .shuffle(imageFiles.length)
    .batch(32)

  return dataset
}

async function trainModel() {
  // 모델 로드
  // const modelUrl = 'https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/5'
  // const model = await tf.loadLayersModel(modelUrl)

  // const model = await mobilenet.load({ version: 2, alpha: 1 });

  const mobilenetModel = await mobilenet.load({ version: 2, alpha: 1 })
  const model = mobilenetModel

  // 데이터셋 로드
  const dataset = loadDataset()

  // 모델 컴파일
  model.compile({
    loss: 'categoricalCrossentropy',
    optimizer: tf.train.adam(0.001),
    metrics: ['accuracy']
  })

  // 모델 학습
  await model.fitDataset(dataset, {
    epochs: 10
  })

  // 학습된 모델 저장
  await model.save('file:///path/to/trained/model')
}

trainModel()
