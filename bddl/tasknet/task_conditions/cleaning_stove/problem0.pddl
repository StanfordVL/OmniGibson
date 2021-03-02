(define (problem cleaning_stove_0)
    (:domain igibson)

    (:objects
     	stove1 - stove
    	sink1 - sink
    	counter1 - counter
    	water1 - water
    	soap1 - soap
    	disinfectant1 - disinfectant
    	washcloth1 - washcloth
    	vegetable_oil1 - vegetable_oil
    	grain1 grain2 grain3 - grain
    	crumb1 crumb2 crumb3 - crumb
    	scum1 scum2 - scum
    	garbage1 - garbage
    	pot1 pot2 - pot
    	pan1 - pan
    )
    
    (:init 
        (and 
            (dusty stove1) 
            (nextto stove1 sink1)
        ) 
        (and 
            (not 
                (scrubbed counter1)
            ) 
            (nextto counter1 sink1)
        ) 
        (inside water1 sink1) 
        (nextto soap1 sink1) 
        (and 
            (nextto disinfectant1 sink1) 
            (nextto disinfectant1 soap1)
        ) 
        (and 
            (not 
                (dusty washcloth1)
            ) 
            (nextto washcloth1 soap1) 
            (nextto washcloth1 sink1)
        ) 
        (ontop vegetable_oil1 stove1) 
        (and 
            (ontop grain1 stove1) 
            (nextto grain2 grain1) 
            (ontop grain2 stove1) 
            (nextto grain3 grain2) 
            (ontop grain3 stove1)
        ) 
        (and 
            (nextto crumb1 grain3) 
            (ontop crumb1 stove1) 
            (nextto crumb2 crumb1) 
            (ontop crumb2 stove1) 
            (nextto crumb3 crumb2) 
            (ontop crumb3 stove1)
        ) 
        (and 
            (ontop scum1 stove1) 
            (nextto scum2 scum1) 
            (ontop scum2 stove1)
        ) 
        (nextto garbage1 sink1) 
        (and 
            (dusty pot1) 
            (ontop pot1 stove1)
        ) 
        (and 
            (nextto pot2 pot1) 
            (dusty pot2) 
            (ontop pot2 stove1)
        ) 
        (and 
            (dusty pan1) 
            (ontop pan1 stove1)
        ) 
        (inroom sink1 kitchen) 
        (inroom counter1 kitchen) 
        (inroom stove1 kitchen)
    )
    
    (:goal 
        (and 
            (scrubbed ?counter1) 
            (scrubbed ?stove1) 
            (inside ?water1 ?sink1) 
            (and 
                (dusty ?washcloth1) 
                (inside ?washcloth1 ?water1) 
                (inside ?washcloth1 ?sink1)
            ) 
            (forall 
                (?vegetable_oil - vegetable_oil) 
                (inside ?vegetable_oil ?water1)
            ) 
            (forall 
                (?grain - grain) 
                (inside ?grain ?garbage1)
            ) 
            (forall 
                (?crumb - crumb) 
                (inside ?crumb ?garbage1)
            ) 
            (forall 
                (?pot - pot) 
                (and 
                    (scrubbed ?pot) 
                    (ontop ?pot ?counter1)
                )
            ) 
            (forall 
                (?pan - pan) 
                (and 
                    (scrubbed ?pan) 
                    (ontop ?pan ?counter1)
                )
            )
        )
    )
)