# Adversarial Machine Learning

### การเรียนรู้ของเครื่องแบบปฏิปักษ์: การก่อกวน และความทนทาน

เมื่อ machine learning เข้ามามีบทบาทในชีวิตจริงเป็นอย่างมากจนกระทั่งความผิดพลาดจากแบบจำลองอาจนำไปสู่ความเสียหายอย่างใหญ่หลวงหรือกระทั่งนำไปสู่การเสียชีวิต การสร้างแบบจำลองทาง machine learning จะคำนึงถึงแต่ความแม่นยำในการตัดสินใจไม่ได้อีกแล้ว เราจำเป็นต้องสนใจ _ความทนทาน_ (robustness) ของแบบจำลองด้วย ซึ่งหมายถึงความสามารถในการตัดสินใจที่มั่นคงเมื่อข้อมูลที่รับเข้ามามีการปนเปื้อนด้วย noise ที่เกิดจากสภาพแวดล้อมหรือแม้กระทั่ง noise ที่เกิดจาก _การก่อกวน_ (perturbation) จากผู้ประสงค์ร้าย

บทความชุดนี้บันทึกความรู้พื้นฐานทางทฤษฎีที่เกี่ยวข้องกับการก่อกวนบนแบบจำลองทาง machine learning ที่ได้รับความนิยมในปัจจุบันเช่น deep learning, การเพิ่มความทนทานให้กับแบบจำลองเพื่อป้องกันการถูกก่อกวน รวมถึงความรู้ทางทฤษฎีในมุมต่าง ๆ ที่เกี่ยวข้องกับความทนทานของแบบจำลองเหล่านี้
โดยเนื้อหาในช่วงแรกอ้างอิงจาก [Adversarial Robustness - Theory and Practice](https://adversarial-ml-tutorial.org) ของ Zico Kolter และ Aleksander Madry

### บทนำ
1. [แบบจำลอง deep learning พอสังเขป](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/intro1)
1. [การสร้าง adversarial example](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/intro2)
1. [การโจมตีแบบกำหนดเป้าหมาย](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/intro3)
1. [Adversarial robustness](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/intro4)
1. [การสร้าง robust classifier](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/intro5)

### การโจมตีและการป้องกัน
1. [แบบจำลอง Linear binary classification](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/attack1)
1. [Adversarial linear binary classification](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/attack2)
1. [Fast Gradient Sign Method](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/attack3)
1. [Projected Gradient Descent](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/attack4)
1. [Adversarial training](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/attack5)

### Robustness Certification
1. [การสร้าง robustness certificate](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/cert1)
1. [Linear programming relaxation](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/cert2)
1. [Quadratic programming](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/cert3)
1. [Semidefinite programming relaxation](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/cert4)

### Robust Features
1. [คุณสมบัติของ adversarial example](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/feat1)
1. [Useful features](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/feat2)
1. [Robust features และ weak features](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/feat3)
1. [การเทรนแบบจำลองด้วย robust features](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/feat4)
1. [ความสามารถของ weak features]()

### การสังเคราะห์ภาพด้วย Robust Classifier
