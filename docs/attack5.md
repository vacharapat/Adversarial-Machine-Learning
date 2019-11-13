{% include lib/mathjax.html %}
# Adversarial training

หลังจากที่เราได้รู้จักวิธีสร้าง adversarial example กันมาแล้ว คราวนี้เราจะกลับมาพิจารณาการสร้างแบบจำลอง machine learning 
ที่มีความทนทานต่อการก่อกวนเหล่านี้ จากที่เราได้ทราบกันมาแล้วว่า สำหรับแบบจำลอง $$h_\theta$$ ใด ๆ หากเราต้องการหาพารามิเตอร์ $$\theta$$
ที่ทำให้แบบจำลองของเราทนทานต่อการก่อกวนมากที่สุด เราจะต้องหาค่า $$\theta$$ ที่นำไปสู่ adverarial risk ที่น้อยที่สุด นั่นคือ เราต้องการค่า $$\theta$$ ที่

$$
\mathbb{E}_{(x,y)\sim\mathcal{D}}[\max_{\delta\in\Delta(x)}\ell(h_\theta(x+\delta),y)]
$$

มีค่าน้อยที่สุด ซึ่งในทางปฏิบัติ เมื่อเรามีชุดข้อมูล $$D=\{(x_1,y_1),\dots,(x_m,y_m)\}$$ เราสนใจการหาค่า $$\theta$$ ที่มีค่าเฉลี่ยของ adversarial loss ของชุดข้อมูลนี้น้อยที่สุด ดังนั้น ปัญหาในการเทรนแบบจำลองของเราสามารถเขียนเป็นปัญหา optimization ได้เป็น

$$
\min_\theta\frac{1}{m}\sum_{i=1}^m \max_{\delta_i\in\Delta(x_i)}\ell(h_\theta(x_i+\delta_i),y_i)
$$

แต่จากที่เราได้ทราบแล้ว การหาค่า $$\theta$$ ที่ต้องการนี้โดยใช้แนวคิดของการทำ gradient descent มีปัญหาตรงที่เราจะหา gradient ของค่าเฉลี่ย adversarial loss นี้ได้ก็ต่อเมื่อเราต้องสามารถแก้ปัญหา maximization ด้านในให้ได้เท่านั้น เนื่องจาก Danskin's theorem ได้แสดงว่า ถ้า

$$
\delta^*=\arg\max_{\delta\in\Delta(x)}\ell(h_\theta(x+\delta),y)
$$

เราจะได้ว่า

$$
\nabla_\theta\max_{\delta\in\Delta(x)}\ell(h_\theta(x+\delta),y) = \nabla_\theta\ell(h_\theta(x+\delta^*),y)
$$

หากเราลองสมมติว่าเราสามารถแก้ปัญหา maximization ด้านในได้ เราก็จะสามารถออกแบบอัลกอริทึมในการเทรนแบบจำลองให้มีความทนทานต่อการก่อกวนโดยใช้เทคนิค stochastic gradient descent (SGD) บน training data $$D$$ ได้ดังนี้

$$
\begin{array}{l}
\text{Repeat:}\\
\quad \text{Select minibatch } B\subseteq D\\
\quad \text{for each sample } (x,y)\in B:\\
\quad \quad \text{Compute } \delta^*(x) = \arg\max_{\delta\in\Delta(x)}\ell(h_\theta(x+\delta),y)\\
\quad \theta\gets\theta -\alpha\frac{1}{|B|}\sum_{(x,y)\in B}\nabla_\theta\ell(h_\theta(x+\delta^*(x)), y)
\end{array}
$$

ในทางปฏิบัติ ถึงแม้ว่าเราไม่สามารถคำนวณหา $$\delta^*(x)$$ ตามต้องการจริง ๆ ได้ เราพบว่าการประมาณ $$\delta^*(x)$$ ด้วยวิธีการเช่น FGSM หรือ PGD นั้นก็ช่วยให้เราได้แบบจำลองที่ทนทานต่อการก่อกวนมากขึ้นจากการเทรนแบบจำลองแบบปกติ ดังนั้น การทำ adversarial training เพื่อสร้างแบบจำลองที่มีความทนทานโดยใช้เทคนิค SGD สามารถสรุปเป็นวิธีการได้ดังนี้

$$
\begin{array}{l}
\text{Repeat:}\\
\quad \text{Select minibatch } B\subseteq D\\
\quad \text{for each sample } (x,y)\in B:\\
\quad \quad \text{Compute adversarial perturbation } \delta^{adv}(x)\\
\quad \theta\gets\theta -\alpha\frac{1}{|B|}\sum_{(x,y)\in B}\nabla_\theta\ell(h_\theta(x+\delta^{adv}(x)), y)
\end{array}
$$

## การทดลอง

จากชุดตัวอย่างข้อมูล MNIST ที่เราใช้ในการทดลองการก่อกวนที่ผ่านมา เมื่อเราลองนำแบบจำลอง CNN มาทำการเทรนใหม่แบบ adversarial training โดยใช้อัลกอริทึม FGSM ในการสร้าง adversarial perturbation ปรากฏว่าแบบจำลองที่ได้มีอัตราความผิดพลาดน้อยกว่า 3% แม้โดนก่อกวนด้วย FGSM หรือ PGD ก็ตาม นั่นหมายความว่าแบบจำลอง CNN ของเรามีความทนทานต่อการก่อกวนมากขึ้นกว่าเดิมเป็นอย่างมาก

<p align="center">
<img width="300" src="https://raw.githubusercontent.com/vacharapat/Adversarial-Machine-Learning/master/images/adv_training.png">
</p>

สังเกตว่า หากเราทำ adversarial training โดยใช้ PGD ในการสร้าง adversarial perturbation ก็อาจได้แบบจำลองที่มีความทนทานมากขึ้นได้ อย่างไรก็ดี เนื่องจากการคำนวณหา adversarial perturbation โดยใช้ PGD แต่ละครั้งนั้นมีการวนรอบทำ gradient descent อยู่ สังเกตว่าหากเราใช้ PGD มาสร้าง adversarial perturbation ในการเทรนนี้ จะทำให้เวลาที่ต้องใช้ในการเทรนเพิ่มขึ้นจากการทำ standard training เป็นอย่างมาก ในขณะที่ FGSM สามารถสร้าง adversarial perturbation ได้ในเวลาอันรวดเร็ว จึงมีความสะดวกมากกว่าในการนำมาทำ adversarial training 
