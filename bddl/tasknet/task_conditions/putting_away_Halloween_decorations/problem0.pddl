(define (problem putting_away_Halloween_decorations_0)
    (:domain igibson)

    (:objects
     	garbage1 - garbage
    	table1 - table
    	tape1 - tape
    	shelf1 - shelf
    	box1 box2 box3 - box
    	gravestone1 gravestone2 - gravestone
    	skeleton1 skeleton2 - skeleton
    	pumpkin1 - pumpkin
    	caldron1 - caldron
    	hat1 - hat
    	candle1 candle2 - candle
    	stake1 stake2 stake3 stake4 - stake
    	sheet1 sheet2 sheet3 - sheet
    )
    
    (:init 
        (nextto garbage1 table1) 
        (ontop tape1 shelf1) 
        (and 
            (under box1 shelf1) 
            (open box1)
        ) 
        (and 
            (under box2 shelf1) 
            (open box2)
        ) 
        (and 
            (under box3 shelf1) 
            (open box3)
        ) 
        (and 
            (under gravestone1 table1) 
            (under gravestone2 table1) 
            (under skeleton1 table1) 
            (under skeleton2 table1) 
            (under pumpkin1 table1) 
            (under caldron1 table1)
        ) 
        (and 
            (ontop hat1 table1) 
            (ontop candle1 table1) 
            (ontop candle2 table1) 
            (ontop stake1 table1) 
            (ontop stake2 table1) 
            (ontop stake3 table1) 
            (ontop stake4 table1) 
            (ontop sheet1 table1) 
            (ontop sheet2 table1) 
            (ontop sheet3 table1)
        )
    )
    
    (:goal 
        (and 
            (forall 
                (?candle - candle) 
                (imply 
                    (broken ?candle) 
                    (inside ?candle ?garbage1)
                )
            ) 
            (exists 
                (?box - box) 
                (forall 
                    (?candle - candle) 
                    (imply 
                        (not 
                            (broken ?candle)
                        ) 
                        (inside ?candle ?box)
                    )
                )
            ) 
            (exists 
                (?box - box) 
                (forall 
                    (?sheet - sheet) 
                    (inside ?sheet ?box)
                )
            ) 
            (exists 
                (?box - box) 
                (forall 
                    (?hat - hat) 
                    (inside ?hat ?box)
                )
            ) 
            (exists 
                (?box - box) 
                (forall 
                    (?stake - stake) 
                    (inside ?stake ?box)
                )
            ) 
            (exists 
                (?box - box) 
                (forall 
                    (?gravestone - gravestone) 
                    (inside ?gravestone ?box)
                )
            ) 
            (exists 
                (?box - box) 
                (forall 
                    (?skeleton - skeleton) 
                    (inside ?skeleton ?box)
                )
            ) 
            (forall 
                (?box - box) 
                (and 
                    (ontop ?box ?shelf1) 
                    (not 
                        (open ?box)
                    )
                )
            ) 
            (inside ?pumpkin1 ?garbage1) 
            (under ?caldron1 ?shelf1)
        )
    )
)