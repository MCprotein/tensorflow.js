import express from 'express'
import * as tf from '@tensorflow/tfjs-node'
import axios from 'axios'

const app = express()
const port = 3000

// 옷 종류 추론

// MobileNet 모델 로드
import * as mobilenet from '@tensorflow-models/mobilenet'
let model: mobilenet.MobileNet

async function loadModel() {
  model = await mobilenet.load()
  console.log('MobileNet 모델이 로드되었습니다.')
}

// 이미지 URL에서 이미지 데이터 로드
async function getImageData(url: string) {
  const response = await axios.get(url, {
    responseType: 'arraybuffer'
  })
  const imageData = new Uint8Array(response.data)
  return imageData
}

async function predictImageColor(url: string) {
  // 이미지 데이터 로드
  const imageData = await getImageData(url)

  console.log(imageData, 'imageData')
  // 이미지 텐서 생성
  const imageTensor = tf.node.decodeImage(imageData)

  // 이미지 크기 변경 및 텐서 모양 변환
  const resizedImage = tf.image.resizeBilinear(imageTensor, [224, 224])
  const batchedImage = resizedImage.expandDims(0) as any

  // 모델에 이미지 전달 및 예측
  // console.log(model)
  const predictions = (await model.classify(batchedImage)) as any[]

  console.log(predictions)
  // 가장 높은 확률의 색상 인덱스 찾기
  const topPredictionIndex = predictions[0].class

  // 색상 레이블 반환
  const colorLabels = [
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
  const predictedColor = colorLabels[topPredictionIndex]

  return predictedColor
}

app.use(express.json())

app.post('/predict', async (req, res) => {
  const { url } = req.body
  const color = await predictImageColor(url)
  res.json({ color })
})

app.listen(port, async () => {
  await loadModel()
  console.log(`서버가 http://localhost:${port} 에서 시작되었습니다.`)
})
