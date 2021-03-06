{% include lib/mathjax.html %}
# Projected Gradient Descent

จากการสร้าง adversarial example ด้วย FGSM สังเกตว่า หากลักษณะความโค้งของ loss function เมื่อเทียบกับ input นั้นไม่เป็น convex
จุดที่มีค่า loss สูงสุดในระยะ $$\epsilon$$ จาก $$x$$ อาจไม่ใช่จุดมุมที่มีทิศทางตรงกับ gradient ของ loss ที่จุด $$x$$ ก็ได้ เนื่องจากทิศทางของ gradient ก็สามารถเปลี่ยนไปได้เมื่อเราเปลี่ยนตำแหน่งไปจากเดิม แม้ว่าจะห่างจากเดิมน้อยมาก

วิธีพื้นฐานในการจัดการปัญหาดังกล่าวก็คือ แทนที่เราจะพิจารณาแค่ gradient ของ loss ที่จุด $$x$$ เพียงจุดเดียว และหาการก่อกวน $$\delta$$ ตามทิศทางของ gradient ดังกล่าวทันที เราเริ่มจากขยับการก่อกวน $$\delta$$ ไปเล็กน้อยตามทิศทางของ gradient ที่จุด $$x$$ (หรือจุด $$x+\delta$$ เมื่อ $$\delta = 0$$) จากนั้นเราทำการคำนวณ gradient ของ loss ที่จุด $$x+\delta$$ ใหม่และขยับ $$\delta$$ ไปตามทิศทางของ gradient ใหม่นี้ไปเรื่อย ๆ สังเกตว่าวิธีการนี้เหมือนกับการทำ gradient descent ในการปรับค่าของพารามิเตอร์ต่าง ๆ ของแบบจำลองในขั้นตอนการเทรนนั่นเอง โดยเราสามารถกำหนดความไวในการขยับค่า $$\delta$$ ได้โดยกำหนดค่า learning rate $$\alpha$$ ดังนั้น ถ้าให้ $$\delta^i$$ เป็นค่า $$\delta$$ หลังการขยับในครั้งที่ $$i$$ เราจะสามารถคำนวณ gradient ของ loss เทียบกับ $$\delta^i$$ ได้จาก

$$
g^i=\nabla_{\delta^i}\ell(h_\theta(x+\delta^i),y)
$$

และการปรับค่า $$\delta$$ ในรอบที่ $$i+1$$ ทำได้โดยการคำนวณ $$\delta^{i+1}$$ จาก $$\delta^i$$ ดังนี้

$$
\delta^{i+1}=\delta^i +\alpha g^i
$$

อย่างไรก็ดี เนื่องจากการก่อกวนที่ยอมรับได้นั้นมีขอบเขตจำกัด กล่าวคือ $$\|\delta\|_\infty\leq\epsilon$$ เราจึงต้องคอยระวังไม่ให้การปรับค่า $$\delta$$ พาเราหลุดออกจากขอบเขตนี้ ซึ่งวิธีหนึ่งที่ทำได้ง่าย ๆ ก็คือ เมื่อใดก็ตามที่ $$\delta$$ มีค่าในบางมิติมากกว่า $$\epsilon$$ เราจะทำการ _clip_ หรือลดค่า $$\delta$$ ในมิติดังกล่าวลงให้เหลือเท่ากับ $$\epsilon$$ เราอาจมองการ clip ค่าของ $$\delta$$ นี้เป็นการ project $$\delta$$ ให้กลับเข้ามาอยู่ในขอบเขตที่ต้องการได้ นั่นคือ เราสามารถเขียนกระบวนการวนรอบปรับค่า $$\delta$$ ได้ดังนี้

$$
\begin{array}{l}
\delta\gets 0\\
\text{Repeat:}\\
\quad g\gets\nabla_\delta\ell(h_\theta(x+\delta),y)\\
\quad \delta\gets\text{Clip}_\epsilon(\delta + \alpha g)
\end{array}
$$

เราเรียกวิธีการนี้ว่า _projected gradient descent_ หรือ PGD รูปด้านล่างแสดงตัวอย่างของการทำ PGD

<p align="center">
<img width="150" src="https://raw.githubusercontent.com/vacharapat/Adversarial-Machine-Learning/master/images/pgd.png">
</p>

## การทดลอง

เมื่อเรานำการโจมตีด้วย PGD มาทดสอบกับแบบจำลอง MLP และ CNN ที่ได้ทดสอบด้วย FGSM ไปแล้ว จะเห็นว่า PGD สามารถเพิ่มอัตราความผิดพลาดของแบบจำลองได้ดีกว่า FGSM โดยสามารถเพิ่มอัตราความผิดพลาดให้กับแบบจำลอง MLP เป็น 96.4% และเพิ่มอัตราความผิดพลาดให้กับแบบจำลอง CNN ได้เป็น 74.3% อย่างไรก็ดี เนื่องจากการโจมตีด้วย PGD นั้นเราต้องทำการวนรอบคำนวณการก่อกวน จึงเสียเวลาทำงานตามจำนวนรอบที่ต้องใช้ ในขณะที่ FGSM นั้นสามารถทำการคำนวณผลลัพธ์ได้ทันที FGSM จึงใช้เวลาทำงานเร็วกว่า PGD

<p align="center">
<img width="300" src="https://raw.githubusercontent.com/vacharapat/Adversarial-Machine-Learning/master/images/pgd_result.png">
</p>

## References
1. [Z. Kolter, A. Madry. Adversarial Robustness - Theory and Practice](https://adversarial-ml-tutorial.org)
1. [A. Madry et al. Towards Deep Learning Models Resistant to Adversarial Attacks. In: International Conference on Learning Representations. 2018](https://arxiv.org/abs/1706.06083)

---
Prev: [Fast Gradient Sign Method](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/attack3)

Next: [Adversarial training](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/attack5)
