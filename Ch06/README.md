# MLiA Ch. 6: support vector machines

## 개요

  Supervised learning, (deep learning에 상대적으로) shallow learning의 대표적인 machine learning 방법이다.

  - 장: generalization 성능 좋다
  - 단: 기본적인 SVM은 이진 & 선형 분류만 지원 (여러 변형 통해 multinomial & 비선형 분류도 가능)

## Training

  - 주어진 training data에 대해 margin을 최대화하는 classifier을 구함
    - Hyperplane: 분류 경계 형성하는 (n-1) 차원의 선형 이진 classifier( $w\cdot x-b=0$ )
    - Support vector: hyperplane에 가장 가까운 점들( $w\cdot x-b=1$, $w\cdot x-b=-1$ )
    - Margin: support vector와 hyperplane 사이의 거리( $\frac{2}{||w||}$ )
      <!-- ![](Svm_max_sep_hyperplane_with_margin.png) -->
  - 다음의 constrained optimization 문제로 귀결됨
    > "Minimize $\|\vec{w}\|$ subject to $y_i(\vec{w}\cdot\vec{x_i} - b) \ge 1$,  for $i = 1,\,\ldots,\,n$"
    ![](assets/README-3d19b.png)
  - Lagrange multiplier 적용(제약 았는 최적화를 제약 없는 최적화 문제로 바꾸어 해결)
    ![](assets/README-9a322.png)
    <!-- ![Support vector와 margin](https://upload.wikimedia.org/wikipedia/commons/2/2a/Svm_max_sep_hyperplane_with_margin.png) -->

## Sequential minimal optimization (SMO) algorithm

  - SVM 구현 알고리즘

  가나다라 마바사아


  ![](https://upload.wikimedia.org/math/a/7/4/a745d413de81c293a28dde584c6717df.png)

  $\vec{w}\cdot\vec{x} - b=-1.\,$

  $$ R_{\mu \nu} - {1 \over 2}g_{\mu \nu}\,R + g_{\mu \nu} \Lambda
  = {8 \pi G \over c^4} T_{\mu \nu} $$

## 참고
  - 'SVM' in Wikipedia: https://ko.wikipedia.org/wiki/서포트_벡터_머신, https://en.wikipedia.org/wiki/Support_vector_machine
  - 'Linear classification' in Standford CS231n (Convolutional Neural Networks for Visual Recognition): http://cs231n.github.io/linear-classify
  - SVM in Stanford CS229 class: http://cs229.stanford.edu/notes/cs229-notes3.pdf
  - OpenCV SVM introduction: http://docs.opencv.org/2.4/doc/tutorials/ml/introduction_to_svm/introduction_to_svm.html
  - 모두를 위한 머신러닝/딥러닝 강의: http://hunkim.github.io/ml/
  - Christopher M. Bishop, Pattern Recognition and Machine Learning (Springer, 2006)
  - 'Entropy' in Wikipedia: https://en.wikipedia.org/wiki/Entropy_%28information_theory%29


-------------------

## Linear classifier (from Stanford class CS231n)

  - Stanford 'CNN for visual recognition' class(http://cs231n.github.io/linear-classify) 참고

### Intro to Linear classification
- Score function: raw data를 class score로 mapping
- Loss function: 예측 score와 ground truth label간 차이를 정량화

### Linear score function
### Interpreting a linear classifier
### Loss function

  - $$L =  \underbrace{ \frac{1}{N} \sum_i L_i }_\text{data loss} + \underbrace{ \lambda R(W) }_\text{regularization loss}$$  
    (or in fully expanded form  
    $$L = \frac{1}{N} \sum_i \sum_{j\neq y_i} \left[ \max(0, f(x_i; W)_j - f(x_i; W)_{y_i} + \Delta) \right] + \lambda \sum_k\sum_l W_{k,l}^2$$ )
  - Hyperparameter (Delta ($$\Delta$$)):
  - Regularization strength (lambda ($$\lambda)):

#### Multiclass SVM

  모든 test 입력애 대한 누적값이 loss. Ground truth의 score가 다른 class의 score보다 hyperparameter 값 이상으로 크지 않은 경우 잘못 분류된 것으로 간주 loss 값 증가, ground truth의 score가 hyperparameter 값 이상으로 나머지 class보다 큰 경우 제대로 분류된 것으로 간주 loss 값 유지.

  - Hinge loss function: 주어진 test 입력의 ground truth class의 score와 나머지 다른 class들의 score 간 차이값(+ hyperparameter)들 중 최대값이 0보다 큰 경우 누적해 나감.

  - Regularization term
    - Loss function >0 만족하는 weight matrix(W)는 다양한 조합 존재
    - W의 값이 커지는 것을 막음
    - 완전히 lineary separable하지 않은 dataset에서 outlier에 대해 영향을 줄임

  - Practical considerations
    - Delta (and lambda) 설정:
    - Binary SVM과 연관성:
    - Optimization in primal: the optimization objectives in their unconstrained primal form
    - Other multicalss SVM formulations: One-Vs-All (OVA) SVM, All-vs-All (AVA) SVM, structured SVM

#### Softmax classifier

  Logistic regression classifier를 multiple classes로 일반화(multinomial logistic regression)한 classifier로 score function의 출력값을 unnormalized log probability로 해석하고 이들의 cross-entropy를 최소화하는 방향으로 training

  - Softmax function
    - 각 class의 score 값($f_{y_i}$)을 normalized probability로 변환
      - Exponentiation: log probability to flattend probability
      - Division by sum: normalization
    - 각 class score 값이 0과 1 사이의 값으로, 그 합은 1이 되도록 변환됨

    $$Softmax(y_i) = e^{f_{y_i}}  / \sum_j e^{f_j}$$

  - Cross-entropy loss function
    - 실제 확률 분포 _p_ 와 예측 확률 분포 _q_ 사이 정의된 loss
      - _p_: 주어진 입력의 correct class에만 1이 할당되는 확률 분포(e.g [0, 0, 1, 0, ... , 0])
      - _q_: softmax function 통해 구한 class i 대한 예측 확률($q = e^{f_{y_i}}  / \sum_j e^{f_j}$)
    - 실제 correct class에 대한 softmax 확률이 높아질수록 cross-entropy 값 낮아짐

    $$H(p,q) = - \sum_x p(x) \log q(x)$$

    > __Information entropy__
    > - 각 symbol의 발생 확률의 negative log 값을 정보량으로 정의
    > - 정보량의 기대값을 entropy로 정의
    > - 각 symbol의 확률이 낮을수록 정보량은 exponentially 증가
    > - Symbol 간 정보량이 균일할수록 entropy는 감소
    >
    > ![](assets/README-39914.png) or
    >
    > ![](assets/README-6dd59.png)

  - 확률적 해석
    - Softmax 적용 결과 = likelihood
      - $P(y_i \mid x_i; W) = \frac{e^{f_{y_i}}}{\sum_j e^{f_j} }$
      - 주어진 $W$, $x_i$에 대해 class i의 조건부 확률
    - Cross-entropy 결과 = negative log likelihood
    - Loss function 최소화 = MLE (maximum likelihood estimation)
      - Regularization term($R(W)$)을 prior로 해석할 수 있으며 이를 포함하는 경우 loss function의 최소화는 MAP(maxium a priori) estimation으로도 해석 가능
      - Regularization strength($\lambda$)가 커질수록 $W$는 작아지면서 확률은 uniform 해짐
    > __Bayes' theorem__
    >   
    > posterior ∝ likelihood × prior

#### SVM vs Softmax

  동일한 형태의 score function($f$, 입력에 대한 weighted sum with bias)을 사용하지면, score에 대한 해석(loss function)에서 차이가 있음

  ![](assets/README-3efd6.png)

  Softmax의 결과가 class의 확률 형태로 나타나 좀 더 해석하기 쉬우며, SVM의 hinge loss의 경우 correct class가 높은 score를 가지기만 하면 다른 class에 대한 score들에 영향을 받지 않는 반면, softmax의 경우 항상 모든 class의 score가 결과가 영향을 받음
