(define (problem cleaning_out_drawers_1)
    (:domain igibson)

    (:objects
     	towel1 towel2 - towel
    	drawer1 drawer2 drawer3 drawer4 drawer5 drawer6 - drawer
    	jean1 jean2 jean3 jean4 - jean
    	pajama1 - pajama
    	shirt1 shirt2 shirt3 shirt4 shirt5 shirt6 - shirt
    	sweater1 sweater2 - sweater
    	sock1 sock2 sock3 sock4 sock5 sock6 - sock
    	pomade1 - pomade
    	watch1 - watch
    	shaver1 - shaver
    	paper_clip1 paper_clip2 paper_clip3 - paper_clip
    	necktie1 necktie2 - necktie
    	underwear1 underwear2 underwear3 - underwear
    	cabinet1 - cabinet
    	shelf1 shelf2 - shelf
    )
    
    (:init 
        (inside towel1 drawer2) 
        (inside towel2 drawer1) 
        (inside jean1 drawer6) 
        (inside jean2 drawer5) 
        (inside jean3 drawer4) 
        (inside jean4 drawer3) 
        (inside pajama1 drawer2) 
        (inside shirt1 drawer1) 
        (inside shirt2 drawer6) 
        (inside shirt3 drawer5) 
        (inside shirt4 drawer4) 
        (inside shirt5 drawer3) 
        (inside shirt6 drawer2) 
        (inside sweater1 drawer1) 
        (inside sweater2 drawer6) 
        (inside sock1 drawer5) 
        (inside sock2 drawer4) 
        (inside sock3 drawer3) 
        (inside sock4 drawer2) 
        (inside sock5 drawer1) 
        (inside sock6 drawer6) 
        (inside pomade1 drawer5) 
        (inside watch1 drawer4) 
        (inside shaver1 drawer3) 
        (inside paper_clip1 drawer2) 
        (inside paper_clip2 drawer1) 
        (inside paper_clip3 drawer6) 
        (inside necktie1 drawer5) 
        (inside necktie2 drawer4) 
        (inside underwear1 drawer3) 
        (inside underwear2 drawer2) 
        (inside underwear3 drawer6) 
        (inside drawer1 cabinet1) 
        (inside drawer2 cabinet1) 
        (inside drawer3 cabinet1) 
        (inside drawer4 cabinet1) 
        (inside drawer5 cabinet1) 
        (inside drawer6 cabinet1) 
        (inside towel1 cabinet1) 
        (inside towel2 cabinet1) 
        (inside jean1 cabinet1) 
        (inside jean2 cabinet1) 
        (inside jean3 cabinet1) 
        (inside jean4 cabinet1) 
        (inside pajama1 cabinet1) 
        (inside shirt1 cabinet1) 
        (inside shirt2 cabinet1) 
        (inside shirt3 cabinet1) 
        (inside shirt4 cabinet1) 
        (inside shirt5 cabinet1) 
        (inside shirt6 cabinet1) 
        (inside sweater1 cabinet1) 
        (inside sweater2 cabinet1) 
        (inside sock1 cabinet1) 
        (inside sock2 cabinet1) 
        (inside sock3 cabinet1) 
        (inside sock4 cabinet1) 
        (inside sock5 cabinet1) 
        (inside sock6 cabinet1) 
        (inside pomade1 cabinet1) 
        (inside watch1 cabinet1) 
        (inside shaver1 cabinet1) 
        (inside paper_clip1 cabinet1) 
        (inside paper_clip2 cabinet1) 
        (inside paper_clip3 cabinet1) 
        (inside necktie1 cabinet1) 
        (inside necktie2 cabinet1) 
        (inside underwear1 cabinet1) 
        (inside underwear2 cabinet1) 
        (inside underwear3 cabinet1) 
        (inroom cabinet1 bedroom) 
        (inroom shelf1 bedroom) 
        (inroom shelf2 bedroom)
    )
    
    (:goal 
        (and 
            (exists 
                (?drawer - drawer) 
                (forall 
                    (?jean - jean) 
                    (inside ?jean ?drawer)
                )
            ) 
            (exists 
                (?drawer - drawer) 
                (forall 
                    (?pajama - pajama) 
                    (inside ?pajama ?drawer)
                )
            ) 
            (exists 
                (?drawer - drawer) 
                (forall 
                    (?shirt - shirt) 
                    (inside ?shirt ?drawer)
                )
            ) 
            (exists 
                (?drawer - drawer) 
                (forall 
                    (?sweater - sweater) 
                    (inside ?sweater ?drawer)
                )
            ) 
            (exists 
                (?drawer - drawer) 
                (forall 
                    (?sock - sock) 
                    (inside ?sock ?drawer)
                )
            ) 
            (ontop ?pomade1 ?shelf1) 
            (ontop ?watch1 ?shelf1) 
            (ontop ?shaver1 ?shelf1) 
            (exists 
                (?shelf - shelf) 
                (forall 
                    (?paper_clip - paper_clip) 
                    (ontop ?paper_clip ?shelf)
                )
            ) 
            (exists 
                (?drawer - drawer) 
                (forall 
                    (?necktie - necktie) 
                    (inside ?necktie ?drawer)
                )
            ) 
            (exists 
                (?drawer - drawer) 
                (forall 
                    (?underwear - underwear) 
                    (inside ?underwear ?drawer)
                )
            ) 
            (exists 
                (?shelf - shelf) 
                (forall 
                    (?towel - towel) 
                    (ontop ?towel ?shelf)
                )
            )
        )
    )
)