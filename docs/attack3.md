{% include lib/mathjax.html %}
# Fast Gradient Sign Method

## จากแบบจำลองเชิงเส้นสู่ deep learning
หลังจากที่เราได้รู้จักการสร้าง adversarial example และการทำ robust optimization ในแบบจำลองที่เป็นเชิงเส้นกันมาแล้ว
เรามาลองพิจารณาบนแบบจำลอง deep neural network กันดูบ้าง ปัญหาแรกที่เราสนใจคือปัญหาการสร้าง adversarial example
สำหรับตัวอย่างข้อมูล $$(x,y)$$ ใด ๆ ซึ่งมองเป็นปัญหา optimization ได้ดังนี้

$$
\max_{\delta\in\Delta}\ell(h_\theta(x+\delta),y)
$$

โดยในคราวนี้เราจะให้ $$h_\theta:\mathbb{R}^n\rightarrow\mathbb{R}^k$$ แทนแบบจำลอง deep neural network ที่มี $$d$$ layer ซึ่งสามารถอธิบายได้ด้วยสมการต่อไปนี้

$$
\begin{split}
z_1&=&x\\
z_{i+1}&=&f_i(W_iz_i+b_i), \text{ สำหรับ } i=1,\dots,d\\
h_\theta(x)&=&z_{d+1}
\end{split}
$$

เมื่อ $$z_i$$ แทนค่าที่จะถูกส่งเข้าไปคำนวณใน layer ที่ $$i$$ ซึ่งมี activation function เป็น $$f_i$$ โดย activation function ที่นิยมในปัจจุบันคือ ReLU $$f_i(z)=\max\{0,z\}$$ สำหรับ layer $$i=1,\dots,d-1$$ และ $$f_d(z)=z$$ สำหรับ parameter ที่ถูกปรับค่าระหว่างการเทรนนั้นคือ $$\theta=\{W_1,b_1,\dots,W_d,b_d\}$$ และ loss function ที่เราใช้ในการเทรนก็คือ softmax cross entropy

$$
\ell(h_\theta(x),y)=\log\sum_{j=1}^ke^{h_\theta(x)_j}-h_\theta(x)_y
$$

ในกรณีของ deep neural network นั้นการแก้ปัญหา maximization สำหรับสร้าง adversarial example ไม่สามารถทำได้ง่ายเหมือนแบบจำลองเชิงเส้น เนื่องจากลักษณะของ loss function เป็น non-convex เราจึงสนใจการประมาณค่าให้ได้ใกล้เคียงที่สุด

## Fast Gradient Sign Method
หนึ่งในอัลกอริทึมกลุ่มแรกที่ถูกเสนอสำหรับสร้าง adversarial example บนแบบจำลอง deep learning นี้เรียกว่า Fast Gradient Sign Method หรือ FGSM

พิจารณาปัญหา maximization เมื่อขอบเขตการก่อกวนอยู่ในรูปของ $$\ell_\infty$$-norm $$\|\delta\|_\infty\leq\epsilon$$ ลักษณะขอบเขตของการก่อกวนสำหรับตัวอย่างข้อมูล $$x$$ แสดงได้ดังรูปด้านล่าง

<p align="center">
<img width="150" src="https://raw.githubusercontent.com/vacharapat/Adversarial-Machine-Learning/master/images/perturbation.png">
</p>

แนวทางพื้นฐานในการแก้ปัญหา optimization ก็คือการทำ gradient descent ซึ่งในกรณีนี้เราต้องการปรับค่า $$\delta$$ ให้ loss มีค่าสูงที่สุด เราต้องเริ่มจากการคำนวณ gradient ของ loss เมื่อเทียบกับ $$\delta$$ ก่อน กล่าวคือเราคำการหา

$$
g=\nabla_\delta\ell(h_\theta(x+\delta),y)
$$

โดยใช้ backpropagation และทำการปรับ $$\delta$$ ไปตามทิศทางของ $$g$$

$$\delta\leftarrow\delta+\alpha g$$

เมื่อ $$\alpha$$ เป็น learning rate อย่างไรก็ดี การปรับ $$\delta$$ เช่นนี้อาจพาเราออกไปนอกขอบเขตของ $$\delta$$ ที่เป็นไปได้หาก $$\alpha$$ มีขนาดใหญ่เกินไป เมื่อเกิดเหตุการณ์ดังกล่าวเราสามารถแก้ปัญหาได้โดยการ project กลับมาให้อยู่ภายในบริเวณที่ต้องการ รูปด้านล่างแสดงตัวอย่างเมื่อ $$\alpha g$$ พาเราออกไปนอกขอบเขต เมื่อเรา project $$\delta$$ ให้กลับมาอยู่ในขอบเขตของ $$\ell_\infty$$-norm จะทำให้ได้ adversarial example เป็นข้อมูลที่อยู่ที่จุดมุมขวาบนของกรอบสี่เหลี่ยม

<p align="center">
<img width="150" src="https://raw.githubusercontent.com/vacharapat/Adversarial-Machine-Learning/master/images/fgsm.png">
</p>

จากตัวอย่างจะเห็นว่า ในกรณีของ $$\ell_\infty$$-norm นี้การ project กลับมาทำได้ง่ายโดยการ _clip_ ค่าของ $$\delta$$ ในแต่ละมิติให้อยู่ในช่วง $$[-\epsilon,\epsilon]$$ และเมื่อ $$\alpha$$ มีขนาดใหญ่พอ เราจะได้ผลอยู่ที่จุดมุมของขอบเขตเสมอ ซึ่งจุดมุมดังกล่าวสามารถหาได้โดยการกำหนดให้ $$\delta_i$$ เป็น $$+\epsilon$$ หรือ $$-\epsilon$$ ตามทิศทางของ $$g_i$$ นั่นคือ

$$
\delta = \epsilon\cdot\text{sign}(g)
$$

เราเรียกกระบวนการสร้าง adversarial example ด้วยวิธีนี้ว่า Fast Gradient Sign Method ซึ่งสามาถคำนวณได้รวดเร็ว เนื่องจากใช้การคำนวณ gradient เพียงครั้งเดียวเท่านั้น

## การทดลอง

ใน [Adversarial Robustness - Theory and Practice](https://adversarial-ml-tutorial.org) ได้ทำการทดลองกับแบบจำลองสองตัว ได้แก่ fully connected multi-layer perceptron (MLP) จำนวน 2 layer และ convolutional neural network (CNN) จำนวน 6 layer กับชุดข้อมูล [MNIST](http://yann.lecun.com/exdb/mnist/)

<p align="center">
<img width="150" src="https://raw.githubusercontent.com/vacharapat/Adversarial-Machine-Learning/master/images/mlp.png">
</p>

<p align="center">
<img width="450" src="https://raw.githubusercontent.com/vacharapat/Adversarial-Machine-Learning/master/images/conv.png">
</p>

รูปด้านล่างแสดงความผิดพลาดของแบบจำลองทั้งคู่ เมื่อทดสอบด้วยชุดข้อมูลทดสอบธรรมดา (clean) และชุดข้อมูลทดสอบที่ถูกก่อกวนด้วย FGSM โดยใช้ $$\epsilon=0.1$$ จะเห็นว่า FGSM สามารถทำให้อัตราผิดพลาดของแบบจำลอง fully connected MLP เพิ่มจาก 2.9% เป็น 92.6% ในขณะที่อัตราผิดพลาดของแบบจำลอง CNN เพิ่มจาก 1.1% เป็น 41.7%

<p align="center">
<img width="250" src="https://raw.githubusercontent.com/vacharapat/Adversarial-Machine-Learning/master/images/fgsm_result.png">
</p>

สังเกตว่าหากแบบจำลองของเราเป็น linear binary classification โดยมีขอบเขตการก่อกวนเป็น $$\ell_\infty$$-norm FGSM จะให้ผลลัพธ์เป็นคำตอบที่ดีที่สุด
อย่างไรก็ดี เรารู้ว่าสำหรับ deep neural network นั้นลักษณะของ loss function ไม่ได้มีทิศทางเป็น linear แม้ในบริเวณเล็ก ๆ ดังนั้นการพุ่งไปยังทิศทางของ gradient โดยตรงในครั้งเดียวนี้อาจไม่ได้พาเรามุ่งหน้าไปหาจุดที่ loss สูงสุดตามต้องการจริง ๆ หากเราต้องการการโจมตีที่ได้ผลดีกว่านี้ เราก็ต้องมีวิธีค้นหาจุดที่ค่า loss สูงที่ดีกว่านี้

## References
1. [Z. Kolter, A. Madry. Adversarial Robustness - Theory and Practice](https://adversarial-ml-tutorial.org)
1. [I. Goodfellow, J. Shlens, C. Szegedy. Explaining and Harnessing Adversarial Examples, In: 3rd International Conference on Learning Representations, 2015](https://arxiv.org/abs/1412.6572)

---
Prev: [Adversarial linear binary classification](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/attack2)

Next: [Projected Gradient Descent](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/attack4)
