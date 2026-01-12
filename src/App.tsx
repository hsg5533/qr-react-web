import React, { useEffect } from "react";
import * as ort from "onnxruntime-web";

import "./App.css";

function App() {
  useEffect(() => {
    const detectionConfidence = 0.6;
    const maskThreshold = 0.9;
    const fixedInferenceFps = 10;
    const modelInputWidth = 384;
    const modelInputHeight = 384;
    const maskColor = { r: 255, g: 0, b: 0, a: 0.45 };
    function sigmoid(x: number) {
      return 1 / (1 + Math.exp(-x));
    }
    function clamp(value: number, min: number, max: number) {
      return Math.max(min, Math.min(max, value));
    }
    const videoElement = document.getElementById("video") as HTMLVideoElement;
    const overlayCanvas = document.getElementById(
      "overlay"
    ) as HTMLCanvasElement;
    const overlayContext = overlayCanvas.getContext("2d", {
      willReadFrequently: true,
    }) as CanvasRenderingContext2D;
    const modelInputCanvas = document.createElement("canvas");
    const modelInputContext = modelInputCanvas.getContext("2d", {
      willReadFrequently: true,
    }) as CanvasRenderingContext2D;
    const maskColorCanvas = document.createElement("canvas");
    const maskColorContext = maskColorCanvas.getContext("2d", {
      willReadFrequently: true,
    }) as CanvasRenderingContext2D;
    let frame: number | null = null;
    let isBusy = false;
    let isRunning = false;
    let lastTimestamp = 0;
    let lastUiUpdate = 0; // 화면 표시 갱신 주기용
    let lastInfer = 0; // 전체(전처리+run+후처리) ms
    let lastRun = 0; // run()만 ms
    let ortSession: ort.InferenceSession | null = null;
    let mediaStream: MediaStream | null = null;
    let modelInputName: string | null = null;
    const clearOverlay = () => {
      overlayContext.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    };
    const stopMediaStream = () => {
      if (mediaStream) {
        for (const track of mediaStream.getTracks()) track.stop();
      }
      mediaStream = null;
    };
    const resizeOverlayToVideo = () => {
      const videoRect = videoElement.getBoundingClientRect();
      const { videoWidth, videoHeight } = videoElement;
      overlayCanvas.style.position = "absolute";
      overlayCanvas.style.width = `${Math.round(videoRect.width)}px`;
      overlayCanvas.style.height = `${Math.round(videoRect.height)}px`;
      overlayCanvas.style.top = `${Math.round(
        videoRect.top + window.scrollY
      )}px`;
      overlayCanvas.style.left = `${Math.round(
        videoRect.left + window.scrollX
      )}px`;
      overlayCanvas.style.pointerEvents = "none";
      if (overlayCanvas.width !== videoWidth) overlayCanvas.width = videoWidth;
      if (overlayCanvas.height !== videoHeight)
        overlayCanvas.height = videoHeight;
    };
    const inferenceLoop = async (timestamp: number) => {
      if (!isRunning) return;
      frame = requestAnimationFrame(inferenceLoop);
      const minDelta = 1000 / Math.max(1, fixedInferenceFps);
      if (isBusy || timestamp - lastTimestamp < minDelta) return;
      lastTimestamp = timestamp;
      isBusy = true;
      let detectionTensor = null;
      let prototypeTensor = null;
      const start = performance.now();
      try {
        const { videoWidth, videoHeight } = videoElement;
        const inferenceFrameWidth = Math.round(
          (videoWidth / videoHeight) * modelInputHeight
        );
        const letterboxScale = Math.round(
          modelInputWidth / inferenceFrameWidth
        );
        const letterboxNewWidth = Math.round(
          inferenceFrameWidth * letterboxScale
        );
        const letterboxNewHeight = Math.round(
          modelInputHeight * letterboxScale
        );
        const letterboxOffsetX = Math.floor(
          (modelInputWidth - letterboxNewWidth) / 2
        );
        const letterboxOffsetY = Math.floor(
          (modelInputHeight - letterboxNewHeight) / 2
        );
        const frameToVideoScaleX = videoWidth / inferenceFrameWidth;
        const frameToVideoScaleY = videoHeight / modelInputHeight;
        const canvasScaleX = overlayCanvas.width / videoWidth;
        const canvasScaleY = overlayCanvas.height / videoHeight;
        modelInputCanvas.width = modelInputWidth;
        modelInputCanvas.height = modelInputHeight;
        modelInputContext.imageSmoothingEnabled = true;
        modelInputContext.fillStyle = "rgb(114,114,114)";
        modelInputContext.fillRect(0, 0, modelInputWidth, modelInputHeight);
        modelInputContext.drawImage(
          videoElement,
          0,
          0,
          videoWidth,
          videoHeight,
          letterboxOffsetX,
          letterboxOffsetY,
          letterboxNewWidth,
          letterboxNewHeight
        );
        const bmp = await createImageBitmap(modelInputCanvas);
        const inputTensor = await ort.Tensor.fromImage(bmp, {
          tensorFormat: "RGB",
          tensorLayout: "NCHW",
          dataType: "float32",
          norm: { mean: 255, bias: 0 },
        });
        bmp.close();
        if (!ortSession || !modelInputName) return;
        const end = performance.now();
        const outputs = await ortSession.run({
          [modelInputName]: inputTensor,
        });
        lastRun = performance.now() - end;
        for (const t of Object.values(outputs)) {
          if (t.dims.length === 3 && t.dims[0] === 1) {
            const N = Math.max(t.dims[1], t.dims[2]);
            const C = Math.min(t.dims[1], t.dims[2]);
            if (N >= 1000 && C >= 8) {
              if (!detectionTensor) detectionTensor = t;
              else {
                const curN = Math.max(
                  detectionTensor.dims[1],
                  detectionTensor.dims[2]
                );
                if (N > curN) detectionTensor = t;
              }
            }
          }
          if (t.dims.length === 4 && t.dims[0] === 1) {
            const nm = t.dims[1];
            const mh = t.dims[2];
            const mw = t.dims[3];
            if (nm >= 8 && mh >= 16 && mw >= 16) {
              if (!prototypeTensor) prototypeTensor = t;
              else {
                const curArea =
                  prototypeTensor.dims[2] * prototypeTensor.dims[3];
                const area = mh * mw;
                if (area > curArea) prototypeTensor = t;
              }
            }
          }
        }
        if (!detectionTensor || !prototypeTensor) return;
        const detectionDims = detectionTensor.dims;
        const prototypeDims = prototypeTensor.dims;
        const detectionData = await detectionTensor.getData();
        const prototypeData = (await prototypeTensor.getData()) as Float32Array;
        detectionTensor.dispose();
        prototypeTensor.dispose();
        const numberOfMaskChannels = prototypeDims[1];
        const dimA = detectionDims[1];
        const dimB = detectionDims[2];
        let getValue: (boxIndex: number, channelIndex: number) => number;
        let numBoxes: number;
        let numChannels: number;
        if (dimA <= dimB) {
          numChannels = dimA;
          numBoxes = dimB;
          getValue = (boxIndex, channelIndex) =>
            (detectionData as Float32Array)[channelIndex * numBoxes + boxIndex];
        } else {
          numBoxes = dimA;
          numChannels = dimB;
          getValue = (boxIndex, channelIndex) =>
            (detectionData as Float32Array)[
              boxIndex * numChannels + channelIndex
            ];
        }
        const classCountWithoutObj = numChannels - 4 - numberOfMaskChannels;
        const classCountWithObj = numChannels - 5 - numberOfMaskChannels;
        const parseVariant = (hasObjectness: boolean) => {
          const classCount = hasObjectness
            ? classCountWithObj
            : classCountWithoutObj;
          if (classCount <= 0) return [];
          const classBaseIndex = hasObjectness ? 5 : 4;
          const coefBaseIndex = classBaseIndex + classCount;
          const boxes = [];
          for (let i = 0; i < numBoxes; i++) {
            let centerX = getValue(i, 0);
            let centerY = getValue(i, 1);
            let width = getValue(i, 2);
            let height = getValue(i, 3);
            const looksNormalized =
              centerX >= 0 &&
              centerX <= 1.5 &&
              centerY >= 0 &&
              centerY <= 1.5 &&
              width >= 0 &&
              width <= 1.5 &&
              height >= 0 &&
              height <= 1.5;
            if (looksNormalized) {
              centerX *= modelInputWidth;
              width *= modelInputWidth;
              centerY *= modelInputHeight;
              height *= modelInputHeight;
            }
            const objectnessRaw = hasObjectness ? getValue(i, 4) : 1.0;
            const objectnessProb =
              objectnessRaw >= 0 && objectnessRaw <= 1
                ? objectnessRaw
                : sigmoid(objectnessRaw);
            let bestClassLogit = -1e9;
            for (let c = 0; c < classCount; c++) {
              const s = getValue(i, classBaseIndex + c);
              if (s > bestClassLogit) bestClassLogit = s;
            }
            const classProb =
              bestClassLogit >= 0 && bestClassLogit <= 1
                ? bestClassLogit
                : sigmoid(bestClassLogit);
            const score = objectnessProb * classProb;
            if (score < detectionConfidence) continue;
            const x1 = centerX - width / 2;
            const y1 = centerY - height / 2;
            const x2 = centerX + width / 2;
            const y2 = centerY + height / 2;
            const maskCoefficients = new Float32Array(numberOfMaskChannels);
            for (let k = 0; k < numberOfMaskChannels; k++) {
              maskCoefficients[k] = getValue(i, coefBaseIndex + k);
            }
            boxes.push({ x1, y1, x2, y2, score, maskCoefficients });
          }
          return boxes;
        };
        const boxesWithoutObj = parseVariant(false);
        const boxesWithObj = parseVariant(true);
        const candidateBoxes =
          boxesWithObj.length > boxesWithoutObj.length
            ? boxesWithObj
            : boxesWithoutObj;
        if (!candidateBoxes.length) {
          clearOverlay();
          return;
        }
        candidateBoxes.sort((a, b) => b.score - a.score);
        const bestBox = candidateBoxes[0];
        const frameBoxX1 = clamp(
          (bestBox.x1 - letterboxOffsetX) / letterboxScale,
          0,
          inferenceFrameWidth
        );
        const frameBoxY1 = clamp(
          (bestBox.y1 - letterboxOffsetY) / letterboxScale,
          0,
          modelInputHeight
        );
        const frameBoxX2 = clamp(
          (bestBox.x2 - letterboxOffsetX) / letterboxScale,
          0,
          inferenceFrameWidth
        );
        const frameBoxY2 = clamp(
          (bestBox.y2 - letterboxOffsetY) / letterboxScale,
          0,
          modelInputHeight
        );
        const roiX0 = Math.max(0, Math.floor(frameBoxX1));
        const roiY0 = Math.max(0, Math.floor(frameBoxY1));
        const roiX1 = Math.min(inferenceFrameWidth, Math.floor(frameBoxX2));
        const roiY1 = Math.min(modelInputHeight, Math.floor(frameBoxY2));
        const roiWidth = roiX1 - roiX0 + 1;
        const roiHeight = roiY1 - roiY0 + 1;
        const drawX = roiX0 * frameToVideoScaleX * canvasScaleX;
        const drawY = roiY0 * frameToVideoScaleY * canvasScaleY;
        const drawW = roiWidth * frameToVideoScaleX * canvasScaleX;
        const drawH = roiHeight * frameToVideoScaleY * canvasScaleY;
        const inputRoiX = letterboxOffsetX + roiX0 * letterboxScale;
        const inputRoiY = letterboxOffsetY + roiY0 * letterboxScale;
        const inputRoiH = roiHeight * letterboxScale;
        const inputRoiW = roiWidth * letterboxScale;
        const protoMaskCount = prototypeDims[1];
        const protoHeight = prototypeDims[2];
        const protoWidth = prototypeDims[3];
        const inputRoiX0 = clamp(Math.floor(inputRoiX), 0, modelInputWidth);
        const inputRoiY0 = clamp(Math.floor(inputRoiY), 0, modelInputHeight);
        const inputRoiX1 = clamp(
          Math.ceil(inputRoiX + inputRoiW),
          0,
          modelInputWidth
        );
        const inputRoiY1 = clamp(
          Math.ceil(inputRoiY + inputRoiH),
          0,
          modelInputHeight
        );
        const inputRoiWidth = inputRoiX1 - inputRoiX0;
        const inputRoiHeight = inputRoiY1 - inputRoiY0;
        const maskLogitThreshold = Math.log(
          maskThreshold / (1 - maskThreshold)
        );
        const protoArea = protoHeight * protoWidth;
        const binaryMask = new Uint8Array(roiWidth * roiHeight);
        const modelInputToProtoScaleX = protoWidth / modelInputWidth;
        const modelInputToProtoScaleY = protoHeight / modelInputHeight;
        const roiToModelInputScaleX = inputRoiWidth / roiWidth;
        const roiToModelInputScaleY = inputRoiHeight / roiHeight;
        for (let roiY = 0; roiY < roiHeight; roiY++) {
          // ROI 픽셀 중심을 modelInput 좌표로
          const modelInputY = inputRoiY0 + (roiY + 0.5) * roiToModelInputScaleY;
          const protoYFloat = modelInputY * modelInputToProtoScaleY;
          const protoY0 = clamp(Math.floor(protoYFloat), 0, protoHeight);
          const protoY1 = clamp(protoY0 + 1, 0, protoHeight);
          const protoYFraction = protoYFloat - Math.floor(protoYFloat);
          const protoYWeight0 = 1 - protoYFraction;
          const protoYWeight1 = protoYFraction;
          const protoRow0BaseIndex = protoY0 * protoWidth;
          const protoRow1BaseIndex = protoY1 * protoWidth;
          for (let roiX = 0; roiX < roiWidth; roiX++) {
            const modelInputX =
              inputRoiX0 + (roiX + 0.5) * roiToModelInputScaleX;
            const protoXFloat = modelInputX * modelInputToProtoScaleX;
            const protoX0 = clamp(Math.floor(protoXFloat), 0, protoWidth - 1);
            const protoX1 = clamp(protoX0 + 1, 0, protoWidth - 1);
            const protoXFraction = protoXFloat - Math.floor(protoXFloat);
            const protoXWeight0 = 1 - protoXFraction;
            const protoXWeight1 = protoXFraction;
            const weight00 = protoXWeight0 * protoYWeight0;
            const weight10 = protoXWeight1 * protoYWeight0;
            const weight01 = protoXWeight0 * protoYWeight1;
            const weight11 = protoXWeight1 * protoYWeight1;
            const protoIndex00 = protoRow0BaseIndex + protoX0;
            const protoIndex10 = protoRow0BaseIndex + protoX1;
            const protoIndex01 = protoRow1BaseIndex + protoX0;
            const protoIndex11 = protoRow1BaseIndex + protoX1;
            let maskLogitValue = 0;
            for (
              let protoMaskChannelIndex = 0;
              protoMaskChannelIndex < protoMaskCount;
              protoMaskChannelIndex++
            ) {
              const protoChannelBaseIndex = protoMaskChannelIndex * protoArea;
              const protoValue00 =
                prototypeData[protoChannelBaseIndex + protoIndex00];
              const protoValue10 =
                prototypeData[protoChannelBaseIndex + protoIndex10];
              const protoValue01 =
                prototypeData[protoChannelBaseIndex + protoIndex01];
              const protoValue11 =
                prototypeData[protoChannelBaseIndex + protoIndex11];
              const bilinearPrototypeValue =
                protoValue00 * weight00 +
                protoValue10 * weight10 +
                protoValue01 * weight01 +
                protoValue11 * weight11;
              maskLogitValue +=
                bestBox.maskCoefficients[protoMaskChannelIndex] *
                bilinearPrototypeValue;
            }
            binaryMask[roiY * roiWidth + roiX] =
              maskLogitValue >= maskLogitThreshold ? 1 : 0;
          }
        }
        const boundary = []; // roi 마스크 경계 외곽선 좌표 배열값
        for (let y = 1; y < roiHeight - 1; y++) {
          const row = y * roiWidth;
          for (let x = 1; x < roiWidth - 1; x++) {
            const idx = row + x;
            if (!binaryMask[idx]) continue;
            const all =
              binaryMask[idx - 1] &
              binaryMask[idx + 1] &
              binaryMask[idx - roiWidth] &
              binaryMask[idx + roiWidth] &
              binaryMask[idx - roiWidth - 1] &
              binaryMask[idx - roiWidth + 1] &
              binaryMask[idx + roiWidth - 1] &
              binaryMask[idx + roiWidth + 1];

            if (all === 0) boundary.push({ x, y });
          }
        }
        boundary.sort((a, b) => (a.x === b.x ? a.y - b.y : a.x - b.x));
        const lower = []; // 아래쪽 경계(왼쪽→오른쪽으로 가는 하단 외곽선)
        for (let i = 0; i < boundary.length; i++) {
          const p = boundary[i];
          while (lower.length >= 2) {
            const o = lower[lower.length - 2];
            const a = lower[lower.length - 1];
            const cross = (a.x - o.x) * (p.y - o.y) - (a.y - o.y) * (p.x - o.x);
            if (cross > 0) break;
            lower.pop();
          }
          lower.push(p);
        }
        const upper = []; // 위쪽 경계(오른쪽→왼쪽으로 돌아오는 상단 외곽선)
        for (let i = boundary.length - 1; i >= 0; i--) {
          const p = boundary[i];
          while (upper.length >= 2) {
            const o = upper[upper.length - 2];
            const a = upper[upper.length - 1];
            const cross = (a.x - o.x) * (p.y - o.y) - (a.y - o.y) * (p.x - o.x);
            if (cross > 0) break;
            upper.pop();
          }
          upper.push(p);
        }
        // 좌측 끝, 우측끝 좌표 중복 방지
        upper.pop();
        lower.pop();
        const hull = lower.concat(upper);
        if (hull.length >= 4) {
          // 3) hull ring + perimeter
          const ring = hull.slice();
          ring.push(hull[0]);

          let per = 0;
          for (let i = 0; i < ring.length - 1; i++) {
            const a = ring[i],
              b = ring[i + 1];
            per += Math.hypot(b.x - a.x, b.y - a.y);
          }
          per = Math.max(1, per);
          // auto-tune eps to get 4 points
          let eps = per * 0.01;
          let simp = null;
          for (let iter = 0; iter < 35; iter++) {
            const n = ring.length;
            const keep = new Uint8Array(n);
            keep[0] = 1;
            keep[n - 1] = 1;
            const stack: number[][] = [];
            stack.push([0, n - 1]);
            while (stack.length) {
              const [s, e] = stack.pop() as number[];
              const ax = ring[s].x;
              const ay = ring[s].y;
              const bx = ring[e].x;
              const by = ring[e].y;
              const vx = bx - ax;
              const vy = by - ay;
              const vv = vx * vx + vy * vy;
              let bestIdx = -1;
              let bestDist = -1;
              for (let i = s + 1; i < e; i++) {
                const px = ring[i].x;
                const py = ring[i].y;
                let dist;
                if (vv <= 1e-9) {
                  dist = Math.hypot(px - ax, py - ay);
                } else {
                  const wx = px - ax;
                  const wy = py - ay;
                  let t = (wx * vx + wy * vy) / vv;
                  if (t < 0) t = 0;
                  else if (t > 1) t = 1;
                  const cx = ax + t * vx;
                  const cy = ay + t * vy;
                  dist = Math.hypot(px - cx, py - cy);
                }
                if (dist > bestDist) {
                  bestDist = dist;
                  bestIdx = i;
                }
              }
              if (bestDist > eps && bestIdx !== -1) {
                keep[bestIdx] = 1;
                stack.push([s, bestIdx]);
                stack.push([bestIdx, e]);
              }
            }
            const out = [];
            for (let i = 0; i < n; i++) if (keep[i]) out.push(ring[i]);
            out.pop(); // drop duplicated last
            simp = out;
            if (simp.length > 4) eps *= 1.25;
            else if (simp.length < 4) eps *= 0.8;
            else break;
          }
          if (!simp) return;
          let qcx = 0;
          let qcy = 0;
          for (let i = 0; i < 4; i++) {
            qcx += simp[i].x;
            qcy += simp[i].y;
          }
          qcx /= 4;
          qcy /= 4;
          const sorted = simp
            .slice()
            .sort(
              (p, q) =>
                Math.atan2(p.y - qcy, p.x - qcx) -
                Math.atan2(q.y - qcy, q.x - qcx)
            );
          let bestIdx = 0;
          let bestVal = Infinity;
          for (let i = 0; i < 4; i++) {
            const v = sorted[i].x + sorted[i].y;
            if (v < bestVal) {
              bestVal = v;
              bestIdx = i;
            }
          }
          const tl = sorted[bestIdx];
          const tr = sorted[(bestIdx + 1) % 4];
          const br = sorted[(bestIdx + 2) % 4];
          const bl = sorted[(bestIdx + 3) % 4];
          const quadOnOverlay = [tl, tr, br, bl].map((p) => {
            const fx = roiX0 + p.x; // inferenceFrame coords
            const fy = roiY0 + p.y;
            return {
              x: fx * frameToVideoScaleX * canvasScaleX,
              y: fy * frameToVideoScaleY * canvasScaleY,
            };
          });
          clearOverlay();
          overlayContext.save();
          overlayContext.beginPath();
          overlayContext.moveTo(quadOnOverlay[0].x, quadOnOverlay[0].y);
          overlayContext.lineTo(quadOnOverlay[1].x, quadOnOverlay[1].y);
          overlayContext.lineTo(quadOnOverlay[2].x, quadOnOverlay[2].y);
          overlayContext.lineTo(quadOnOverlay[3].x, quadOnOverlay[3].y);
          overlayContext.closePath();
          overlayContext.strokeStyle = "rgba(0,255,0,1)";
          overlayContext.lineWidth = 4;
          overlayContext.stroke();
          overlayContext.restore();
        }
        maskColorCanvas.width = roiWidth;
        maskColorCanvas.height = roiHeight;
        const maskColorImageData = maskColorContext.createImageData(
          roiWidth,
          roiHeight
        );
        const out = maskColorImageData.data;
        for (let i = 0; i < roiWidth * roiHeight; i++) {
          const p = i * 4;
          if (binaryMask[i]) {
            out[p] = maskColor.r;
            out[p + 1] = maskColor.g;
            out[p + 2] = maskColor.b;
            out[p + 3] = Math.max(
              0,
              Math.min(255, Math.round((maskColor.a ?? 0.45) * 255))
            );
          } else {
            out[p] = 0;
            out[p + 1] = 0;
            out[p + 2] = 0;
            out[p + 3] = 0;
          }
        }
        // maskColorContext.putImageData(maskColorImageData, 0, 0);
        overlayContext.save();
        overlayContext.imageSmoothingEnabled = true;
        // overlayContext.drawImage(maskColorCanvas, drawX, drawY, drawW, drawH);
        overlayContext.fillStyle = "rgba(0,0,0,0.95)";
        overlayContext.font = `900 ${Math.round(
          Math.min(overlayCanvas.width, overlayCanvas.height) * 0.06
        )}px system-ui`;
        overlayContext.textAlign = "center";
        overlayContext.textBaseline = "middle";
        overlayContext.fillText(
          `${Math.round(bestBox.score * 100)}%`,
          drawX + drawW / 2,
          drawY + drawH / 2
        );
        overlayContext.restore();
      } catch (error) {
        if (error instanceof Error) {
          console.error(error.message);
        }
      } finally {
        isBusy = false;
        lastInfer = performance.now() - start;
        // 200ms마다 화면 표시 업데이트
        if (timestamp - lastUiUpdate > 200) {
          lastUiUpdate = timestamp;
          //  화면에 텍스트로 표시
          const status = document.getElementById("status");
          if (status) {
            status.textContent = `사이클 ${lastInfer.toFixed(
              1
            )}ms / 추론 ${lastRun.toFixed(1)}ms`;
          }
        }
      }
    };
    document.getElementById("start")!!.addEventListener("click", async () => {
      try {
        if (!ortSession) {
          ortSession = await ort.InferenceSession.create("model/best.onnx", {
            executionProviders: ["webgl", "wasm"],
            graphOptimizationLevel: "all",
          });
          modelInputName = ortSession.inputNames[0];
        }
        stopMediaStream();
        mediaStream = await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: {
            facingMode: { ideal: "environment" },
            width: { ideal: 1920 },
            height: { ideal: 1080 },
            frameRate: { ideal: 60 },
          },
        });
        videoElement.srcObject = mediaStream;
        await videoElement.play();
        resizeOverlayToVideo();
        window.addEventListener("scroll", resizeOverlayToVideo, {
          passive: true,
        });
        window.addEventListener("resize", resizeOverlayToVideo, {
          passive: true,
        });
        window.addEventListener("orientationchange", resizeOverlayToVideo, {
          passive: true,
        });
        isBusy = false;
        isRunning = true;
        lastTimestamp = 0;
        if (frame) cancelAnimationFrame(frame);
        frame = requestAnimationFrame(inferenceLoop);
      } catch (error) {
        if (error instanceof Error) {
          console.error(error.message);
        }
      }
    });
    document.getElementById("stop")!!.addEventListener("click", () => {
      isRunning = false;
      if (frame) cancelAnimationFrame(frame);
      frame = null;
      stopMediaStream();
      videoElement.srcObject = null;
      clearOverlay();
    });
    document.getElementById("switch")!!.addEventListener("click", async () => {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const cams = devices.filter((device) => device.kind === "videoinput");
      let camIndex = 0;
      if (!mediaStream) return;
      const curId = mediaStream.getVideoTracks()[0].getSettings().deviceId;
      if (curId) {
        const idx = cams.findIndex((cam) => cam.deviceId === curId);
        if (idx >= 0) camIndex = (idx + 1) % cams.length;
      }
      stopMediaStream();
      videoElement.srcObject = null;
      clearOverlay();
      mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          deviceId: { exact: cams[camIndex].deviceId }, // ✅ 핵심: deviceId 사용
          width: { ideal: 1920 },
          height: { ideal: 1080 },
          frameRate: { ideal: 60 },
        },
      });
      videoElement.srcObject = mediaStream; // ✅ 새 스트림을 video에 연결
      await videoElement.play();
      resizeOverlayToVideo();
    });
  }, []);

  return (
    <div className="App">
      <video id="video" playsInline autoPlay muted></video>
      <canvas id="overlay"></canvas>
      <div className="hud-top">
        <button id="start">시작</button>
        <button id="stop">정지</button>
      </div>
    </div>
  );
}

export default App;
