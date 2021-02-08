(define (problem clearing_the_table_after_dinner_0)
    (:domain igibson)

    (:objects
     	table1 - table
    	cutlery1 cutlery2 cutlery3 cutlery4 cutlery5 - cutlery
    	cup1 cup2 cup3 cup4 - cup
    	garbage1 - garbage
    	dishwasher1 - dishwasher
    	flatware1 flatware2 flatware3 flatware4 flatware5 - flatware
    	condiment1 condiment2 condiment3 condiment4 condiment5 - condiment
    	crumb1 crumb2 crumb3 crumb4 crumb5 crumb6 - crumb
    )
    
    (:init 
        (not 
            (scrubbed table1)
        ) 
        (ontop cutlery5 table1) 
        (ontop cutlery4 table1) 
        (ontop cutlery3 table1) 
        (ontop cutlery2 table1) 
        (ontop cutlery1 table1) 
        (ontop cup4 table1) 
        (ontop cup3 table1) 
        (ontop cup2 table1) 
        (ontop cup1 table1) 
        (nextto garbage1 dishwasher1) 
        (ontop flatware5 table1) 
        (ontop flatware4 table1) 
        (ontop flatware3 table1) 
        (ontop flatware2 table1) 
        (ontop flatware1 table1) 
        (and 
            (ontop condiment1 flatware1) 
            (under table1 condiment1)
        ) 
        (and 
            (ontop condiment2 flatware2) 
            (under table1 condiment2)
        ) 
        (and 
            (ontop condiment3 flatware3) 
            (under table1 condiment3)
        ) 
        (and 
            (ontop condiment4 flatware4) 
            (under table1 condiment4)
        ) 
        (and 
            (ontop condiment5 flatware5) 
            (under table1 condiment5)
        ) 
        (and 
            (ontop crumb1 flatware1) 
            (under table1 crumb1)
        ) 
        (and 
            (ontop crumb2 flatware2) 
            (under table1 crumb2)
        ) 
        (and 
            (ontop crumb3 flatware3) 
            (under table1 crumb3)
        ) 
        (and 
            (ontop crumb4 flatware4) 
            (under table1 crumb4)
        ) 
        (and 
            (ontop crumb5 flatware5) 
            (under table1 crumb5)
        ) 
        (ontop crumb6 table1) 
        (inroom table1 diningroom) 
        (inroom dishwasher1 kitchen)
    )
    
    (:goal 
        (and 
            (forall 
                (?crumb - crumb) 
                (inside ?crumb ?garbage1)
            ) 
            (forall 
                (?flatware - flatware) 
                (imply 
                    (and 
                        (not 
                            (ontop ?crumb ?flatware)
                        ) 
                        (not 
                            (ontop ?condiment ?flatware)
                        )
                    ) 
                    (inside ?flatware ?dishwasher1)
                )
            ) 
            (forall 
                (?condiment - condiment) 
                (inside ?condiment ?garbage1)
            ) 
            (scrubbed ?table1) 
            (forall 
                (?cup - cup) 
                (inside ?cup ?dishwasher1)
            ) 
            (forall 
                (?cutlery - cutlery) 
                (inside ?cutlery ?dishwasher1)
            )
        )
    )
)