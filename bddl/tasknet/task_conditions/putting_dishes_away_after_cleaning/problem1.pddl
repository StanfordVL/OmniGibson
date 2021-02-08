(define (problem putting_dishes_away_after_cleaning_1)
    (:domain igibson)

    (:objects
     	counter1 - counter
    	sink1 - sink
    	cabinet1 - cabinet
    	bowl1 bowl2 bowl3 bowl4 - bowl
    	plate1 plate2 plate3 plate4 - plate
    	fork1 fork2 fork3 fork4 - fork
    	knife1 knife2 knife3 knife4 - knife
    	shelf1 - shelf
    )
    
    (:init 
        (scrubbed counter1) 
        (scrubbed sink1) 
        (open cabinet1) 
        (ontop bowl1 counter1) 
        (ontop bowl2 counter1) 
        (ontop bowl3 counter1) 
        (ontop bowl4 counter1) 
        (scrubbed bowl1) 
        (scrubbed bowl2) 
        (scrubbed bowl3) 
        (scrubbed bowl4) 
        (ontop plate1 counter1) 
        (ontop plate2 counter1) 
        (ontop plate3 counter1) 
        (ontop plate4 counter1) 
        (scrubbed plate1) 
        (scrubbed plate2) 
        (scrubbed plate3) 
        (scrubbed plate4) 
        (inside fork1 sink1) 
        (inside fork2 sink1) 
        (inside fork3 sink1) 
        (inside fork4 sink1) 
        (scrubbed fork1) 
        (scrubbed fork2) 
        (scrubbed fork3) 
        (scrubbed fork4) 
        (inside knife1 sink1) 
        (inside knife2 sink1) 
        (inside knife3 sink1) 
        (inside knife4 sink1) 
        (scrubbed knife1) 
        (scrubbed knife2) 
        (scrubbed knife3) 
        (scrubbed knife4) 
        (inroom cabinet1 kitchen) 
        (inroom sink1 kitchen) 
        (inroom counter1 kitchen) 
        (inroom shelf1 kitchen)
    )
    
    (:goal 
        (and 
            (forall 
                (?bowl - bowl) 
                (ontop ?bowl ?shelf1)
            ) 
            (forall 
                (?plate - plate) 
                (ontop ?plate ?shelf1)
            ) 
            (forall 
                (?fork - fork) 
                (inside ?fork ?cabinet1)
            ) 
            (forall 
                (?knife - knife) 
                (inside ?knife ?cabinet1)
            )
        )
    )
)