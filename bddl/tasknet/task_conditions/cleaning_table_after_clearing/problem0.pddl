(define (problem cleaning_table_after_clearing_0)
    (:domain igibson)

    (:objects
     	table.n.02_1 - table.n.02
    	soap.n.01_1 - soap.n.01
    	water.n.06_1 - water.n.06
    	sink.n.01_1 - sink.n.01
    	floor.n.01_1 - floor.n.01
    )
    
    (:init 
        (stained table.n.02_1) 
        (ontop soap.n.01_1 table.n.02_1) 
        (inside water.n.06_1 sink.n.01_1) 
        (inroom table.n.02_1 dining_room) 
        (inroom floor.n.01_1 dining_room) 
        (inroom sink.n.01_1 kitchen)
    )
    
    (:goal 
        (and 
            (not 
                (stained ?table.n.02_1)
            ) 
            (inside ?water.n.06_1 ?sink.n.01_1) 
            (inside ?soap.n.01_1 ?sink.n.01_1)
        )
    )
)