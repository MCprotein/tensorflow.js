import * as tf from '@tensorflow/tfjs-node'
import * as mobilenet from '@tensorflow-models/mobilenet'
import axios from 'axios'
import express from 'express'

const app = express()
const port = 3000

let model: mobilenet.MobileNet

async function loadModel() {
  model = await mobilenet.load()
  console.log('MobileNet 모델이 로드되었습니다.')
}

async function getImageData(url: string) {
  const response = await axios.get(url, {
    responseType: 'arraybuffer'
  })
  const imageData = new Uint8Array(response.data)
  return imageData
}

async function predictImage(url: string) {
  const imageData = await getImageData(url)
  const decodedImage = tf.node.decodeImage(imageData)
  const image: tf.Tensor3D = tf.image.resizeBilinear(decodedImage, [224, 224])
  const logits = model.infer(image)
  const classes = await model.classify(image)
  return classes[0].className
}

app.use(express.json())

app.post('/predict', async (req, res) => {
  const { url } = req.body
  const prediction = await predictImage(url)
  res.json({ prediction })
})

app.listen(port, async () => {
  await loadModel()
  console.log(`서버가 http://localhost:${port} 에서 시작되었습니다.`)
})
