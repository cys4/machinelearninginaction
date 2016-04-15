# MLiA - Ch. 06: support vector machines

## SVM 개요
* 기본적으로 이진 & 선형 분류 => 여러 변형을 통하여 multinomial & 비선형 분류도 가능
* (Deep learning에 상대적으로) shallow learning의 대표적인 방법
* 장: generalization 성능 좋다 vs. 단: 기본적인 SVM은 이진 분류만 지원

### 특징
* Support vector: hyperplane(분류 결정 경계)에 가장 가까운 점들
* Margin: support vector와 hyperplane 사이의 거리
* 최적화: margin을 최대화하는 hyperplane을 구함 => 제한된 tranining data로부터 robust한 classifier를 얻어냄
