import React, {useEffect, useRef, useState} from 'react';
import { initNotifications, notify } from '@mycv/f8-notification';
import './App.css';
import {Howl} from 'howler';
import * as mobilenet from '@tensorflow-models/mobilenet';
import * as knnClassifier from '@tensorflow-models/knn-classifier';
import '@tensorflow/tfjs-backend-webgl';

import soundURL from './assets/oi-ban-oi.mp3';

var sound = new Howl({
  src: [soundURL]
});

const NOT_TOUCH_LABEL = "not_touch";
const TOUCHED_LABEL = "touched";
const TRAINING_TIMES = 50;
const TOUCHED_CONFIDENCE = 0.8;

function App() {
  const video = useRef();
  const classifier = useRef();
  const canPlaySound = useRef(true);
  const mobilenetModule = useRef();
  const [touched, setTouched] = useState(false);
  const [message, setMessage] = useState('');
  const [step, setStep] = useState(0);

  const init = async () => {
    setMessage('Khởi động ứng dụng...');
    console.log('init...');
    await setupCamera();
    setMessage('Thiết lập camera thành công.');
    console.log('setup camera success');

    classifier.current = knnClassifier.create();
    mobilenetModule.current = await mobilenet.load();

    initNotifications({ cooldown: 3000 });
  };

  const setupCamera = () => {
    return new Promise((resolve, reject) => {
      navigator.getUserMedia = navigator.getUserMedia || 
      navigator.webkitGetUserMedia ||
      navigator.mozGetUserMedia ||
      navigator.msGetUserMedia;

      if (navigator.getUserMedia) {
        navigator.getUserMedia(
          {video: true },
          stream => {
            video.current.srcObject = stream;
            video.current.addEventListener('loadeddata', resolve);
          },
          error => reject(error)
        );
      } else {
        reject();
      }
    });
  };

  const train = async label => {
    console.log(`[${label}] Đang train trong máy...`);
    for (let i = 0; i < TRAINING_TIMES; ++i) {
      setMessage(`Đang đào tạo ${label === NOT_TOUCH_LABEL ? 'bước 1 (Không chạm)' : 'bước 2 (Chạm)'}: ${parseInt((i+1) / TRAINING_TIMES * 100)}%`);
      console.log(`Tiến độ ${parseInt((i+1) / TRAINING_TIMES * 100)}%`);
      await training(label);
    }
    setStep(step + 1);
  };

  const training = label => {
    return new Promise(async resolve => {
      const embedding = mobilenetModule.current.infer(
        video.current, 
        true
      );
      classifier.current.addExample(embedding, label);
      await sleep(100);
      resolve();  
    });
  };

  const runLoop = async () => {
    while (true) {
      const embedding = mobilenetModule.current.infer(
        video.current,
        true
      );
      const result = await classifier.current.predictClass(embedding);

      if (result.label === TOUCHED_LABEL && result.confidences[result.label] > TOUCHED_CONFIDENCE) {
        if (canPlaySound.current) {
          canPlaySound.current = false;
          sound.play();
        }
        notify('Bỏ tay ra', { body: 'Bạn vừa chạm tay vào mặt' });
        setTouched(true);
      } else {
        setTouched(false);
      }
      await sleep(200);
    }
  };

  const sleep = (ms = 0) => {
    return new Promise(resolve => setTimeout(resolve, ms));
  };

  useEffect(() => {
    init();
    sound.on('end', function() {
      canPlaySound.current = true;
    });
  }, []);

  return (
    <div className={`main ${touched ? 'touched' : ''}`}>
      <video
        ref={video}
        className="video"
        autoPlay
      ></video>
      <div className="message">{message}</div>
      {step === 0 && <button className="btn" onClick={() => train(NOT_TOUCH_LABEL)}>Bắt đầu Bước 1: Không chạm</button>}
      {step === 1 && <button className="btn" onClick={() => train(TOUCHED_LABEL)}>Bắt đầu Bước 2: Chạm</button>}
      {step === 2 && <button className="btn" onClick={() => { setMessage('Bắt đầu phát hiện...'); runLoop(); }}>Chạy Mô hình</button>}
    </div>
  );
}

export default App;
