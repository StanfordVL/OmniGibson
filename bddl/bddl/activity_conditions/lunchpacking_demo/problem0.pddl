(define (problem packlunch)
    (:domain igibson)
    (:objects
        cereal1 cereal2 cereal3 cereal4 - cereal
        chip1 chip2 chip3 chip4 - chip        
        fruit1 fruit2 fruit3 fruit4 - fruit
        snacks1 snacks2 snacks3 snacks4 - snacks
        soda1 soda2 soda3 soda4 - soda
        eggs1 eggs2 eggs3 eggs4 - eggs
        dish1 dish2 dish3 dish4 - dish
        top_cabinet1 top_cabinet2 - top_cabinet
        counter1 counter2 - counter
    )
    (:goal
        (and
            (forall (?dish - dish) (exists (?cereal - cereal) (inside ?cereal ?dish)))
            (forall (?dish - dish) (exists (?chip - chip) (inside ?chip ?dish)))
            (forall (?dish - dish) (exists (?fruit - fruit) (inside ?fruit ?dish)))
            (forall (?dish - dish) (exists (?snacks - snacks) (inside ?snacks ?dish)))
            (forall (?dish - dish) (exists (?soda - soda) (inside ?soda ?dish)))
            (forall (?dish - dish) (exists (?eggs - eggs) (inside ?eggs ?dish)))
        )
    )
    (:init
        (inside chip1 top_cabinet1)
        (inside chip2 top_cabinet1)
        (inside chip3 top_cabinet1)
        (inside chip4 top_cabinet1)
        (inside snacks1 top_cabinet1)
        (inside snacks2 top_cabinet1)
        (inside snacks3 top_cabinet1)
        (inside snacks4 top_cabinet1)
        (inside eggs1 top_cabinet2)
        (inside eggs2 top_cabinet2)
        (inside eggs3 top_cabinet2)
        (inside eggs4 top_cabinet2)
        (inside soda1 top_cabinet2)
        (inside soda2 top_cabinet2)
        (inside soda3 top_cabinet2)
        (inside soda4 top_cabinet2)
        (onTop cereal1 counter1)
        (onTop cereal2 counter1)
        (onTop cereal3 counter1)
        (onTop cereal4 counter1)
        (onTop fruit1 counter2)
        (onTop fruit2 counter2)
        (onTop fruit3 counter2)
        (onTop fruit4 counter2)
    )
)
