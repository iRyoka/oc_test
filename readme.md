Дорогие коллеги, 

спасибо, что уделили время моему скромному труду. Я получил большое удовольствие,
решая эти задачи многому научился в процессе. Прошу прощения, что не положил сюда использованные пакеты, но они абсолютно стандартны, а многие из них достаточно тяжелые.
Тот же torch с cuda весит 2Гб, без -- 200Мб. Я указал использованные мной версии. Почти все они -- из свежей anaconda. Если возникнут проблемы, я пришлю нужный пакте в нужной версии. 

Структура файлов в этом репозитории такова:

    modules.txt                                 - список версий, использованных пакетов
    segmentation1.ipynb                         - решение задачи 1
     data1.py                                   
     modeltools1.py
     unet.py
     filters1.py

     weights-FocalLoss-30000epochs.dat          - веса к задаче 1
     weights_topography_HSVJITNORM_10000.dat

     prob_demo1.png                             - иллюстрации к задаче 1
     prob_demo2.png
     prob_demo3.png
     
    lungs.ipynb                                 - решение задачи 2
     lungsdetector.py
     data2.py
     modeltools2.py

     lungs_weights100.dat                       - веса для задачи 2
     lungs_weights300.dat

     lungs_demo1.png                            - иллюстрации к задаче 2
     lungs_demo2.png
     lungs_demo3.png
     lungs_demo4.png

    problem3.md                                 - мысли по задаче 3, которую я, 
                                                  увы, не успел решить

    stitching.ipynb                             - решение задачи 4
     data4.py
     stitcher.py

     stitched_1.jpg                             - иллюстрации к задаче 4
     stitched_LOCAL.jpg
     stitched_LOCAL_AND_GLOBAL.jpg
     stitched_NONE.jpg
     stitched_wrong_gamma.jpg
