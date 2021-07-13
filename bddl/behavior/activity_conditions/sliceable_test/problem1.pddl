(define (problem packlunch)
    (:domain igibson)
    (:objects
        table.n.02_1 - table.n.02
        apple.n.01_1 - apple.n.01
        apricot.n.02_1 - apricot.n.02
        chestnut.n.03_1 - chestnut.n.03
        coconut.n.01_1 - coconut.n.01
        kiwi.n.03_1 - kiwi.n.03
        lemon.n.01_1 - lemon.n.01
        mushroom.n.05_1 - mushroom.n.05
        orange.n.01_1 - orange.n.01
        peach.n.03_1 - peach.n.03
        pear.n.01_1 - pear.n.01
        pineapple.n.02_1 - pineapple.n.02
        plum.n.02_1 - plum.n.02
        pomegranate.n.02_1 - pomegranate.n.02
        pomelo.n.02_1 - pomelo.n.02
        strawberry.n.01_1 - strawberry.n.01
        tomato.n.01_1 - tomato.n.01
        vidalia_onion.n.01_1 - vidalia_onion.n.01
        carving_knife.n.01_1 - carving_knife.n.01
    )
    (:init
        (inRoom table.n.02_1 living_room)
        (onTop apple.n.01_1 table.n.02_1)
        (onTop apricot.n.02_1 table.n.02_1)
        (onTop chestnut.n.03_1 table.n.02_1)
        (onTop coconut.n.01_1 table.n.02_1)
        (onTop kiwi.n.03_1 table.n.02_1)
        (onTop lemon.n.01_1 table.n.02_1)
        (onTop mushroom.n.05_1 table.n.02_1)
        (onTop orange.n.01_1 table.n.02_1)
        (onTop peach.n.03_1 table.n.02_1)
        (onTop pear.n.01_1 table.n.02_1)
        (onTop pineapple.n.02_1 table.n.02_1)
        (onTop plum.n.02_1 table.n.02_1)
        (onTop pomegranate.n.02_1 table.n.02_1)
        (onTop pomelo.n.02_1 table.n.02_1)
        (onTop strawberry.n.01_1 table.n.02_1)
        (onTop tomato.n.01_1 table.n.02_1)
        (onTop vidalia_onion.n.01_1 table.n.02_1)
        (onTop carving_knife.n.01_1 table.n.02_1)
    )
    (:goal
        (and 
            (sliced apple.n.01_1)
            (sliced apricot.n.02_1)
            (sliced chestnut.n.03_1)
            (sliced coconut.n.01_1)
            (sliced kiwi.n.03_1)
            (sliced lemon.n.01_1)
            (sliced mushroom.n.05_1)
            (sliced orange.n.01_1)
            (sliced peach.n.03_1)
            (sliced pear.n.01_1)
            (sliced pineapple.n.02_1)
            (sliced plum.n.02_1)
            (sliced pomegranate.n.02_1)
            (sliced pomelo.n.02_1)
            (sliced strawberry.n.01_1)
            (sliced tomato.n.01_1)
            (sliced vidalia_onion.n.01_1)
        )
    )
)