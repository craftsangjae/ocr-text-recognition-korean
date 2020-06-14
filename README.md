Text Recognition Model From Scratch
---

### Objective

OCR Text Recognition은 단어 혹은 문장 이미지를 읽어 텍스트 글자를 반환하는 알고리즘을 통칭합니다. Computer vision에서 산업 내 가치가 뛰어난 기술 중 하나입니다. 하지만 아직 한글에 대한 학습은 생각보다 성능이 아직 나오지 않고 있습니다. 그래서 이 문제를 해결하기 위해 기존 연구에 한글 고유의 특성한 `JamoEmbedding`을 활용하여 풀어가는 과정을 담았습니다. 아래는 리파짓토리에서 다루는 데이터셋의 형태입니다. 

<img width="370" alt="스크린샷 2020-06-14 오전 11 04 42" src="https://user-images.githubusercontent.com/66022630/84583038-e57b1980-ae2e-11ea-8858-97a5dd6f7740.png">


### 환경

tensorflow 2.0으로 작성되어 있으므로, TF 1.0대의 경우 정상적으로 돌아가지 않을 수 있습니다.

### 디렉토리 구조

현재 어떤 식으로 구현되어 있는 지는 `scripts/` 아래의 폴더를 참고하시면 됩니다. <br>
````markdown
models/
   |- generator.py : Keras의 `Data Generator` 클래스가 구현된 스크립트 
   |- jamo.py : 한글 자모자를 다루는 메소드들이 구현된 스크립트
   |- layers.py : Text Recognition Model에 관련된 custom Layer들이 구현된 스크립트
   |- losses.py : Text Recognition Model에 관련된 Custom Losses들이 구현된 스크립트 
   |- optimizer.py : Custom Optimizer, ADAM이 구현된 스크립트
   |- utils.py : 기타 시각화 혹은 generator에 이용되는 스크립트
````

### 구성

실행 관련된 코드들은 `scripts/`에 수록되어 있습니다.

1. CRNN 
2. SRN : (Seq2Seq + Attention)
3. 한글 Text Recognition Model (JamoEmbedding)

