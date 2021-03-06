
type expr =
  | VarX
  | VarY
  | Sine of expr
  | Cosine of expr
  | Average of expr* expr
  | Times of expr* expr
  | Thresh of expr* expr* expr* expr;;

let buildAverage (e1,e2) = Average (e1, e2);;

let buildCosine e = Cosine e;;

let buildSine e = Sine e;;

let buildThresh (a,b,a_less,b_less) = Thresh (a, b, a_less, b_less);;

let buildTimes (e1,e2) = Times (e1, e2);;

let buildX () = VarX;;

let buildY () = VarY;;

let rec build (rand,depth) =
  if depth < 1
  then
    let base = rand (0, 2) in
    match base with
    | 0 -> buildX ()
    | 1 -> buildY ()
    | _ -> (if base < 0 then buildX () else buildY ())
  else
    (let recurse = rand (0, 5) in
     match recurse with
     | 0 -> buildSine (build (rand, (depth - 1)))
     | 1 -> buildCosine (build (rand, (depth - 1)))
     | 2 ->
         buildAverage
           ((build (rand, (depth - 1))), (build (rand, (depth - 1))))
     | 3 ->
         buildTimes
           ((build (rand, (depth - 1))), (build (rand, (depth - 1))))
     | 4 ->
         buildThresh
           ((build (rand, (depth - 1))), (build (rand, (depth - 1))),
             (build (rand, (depth - 1))), (build (rand, (depth - 1))))
     | _ ->
         if recurse > 2
         then buildCosine (rand, (depth - 1))
         else buildSine (rand, (depth - 1)));;


(* fix

type expr =
  | VarX
  | VarY
  | Sine of expr
  | Cosine of expr
  | Average of expr* expr
  | Times of expr* expr
  | Thresh of expr* expr* expr* expr;;

let buildAverage (e1,e2) = Average (e1, e2);;

let buildCosine e = Cosine e;;

let buildSine e = Sine e;;

let buildThresh (a,b,a_less,b_less) = Thresh (a, b, a_less, b_less);;

let buildTimes (e1,e2) = Times (e1, e2);;

let buildX () = VarX;;

let buildY () = VarY;;

let rec build (rand,depth) =
  if depth < 1
  then
    let base = rand (0, 2) in
    match base with
    | 0 -> buildX ()
    | 1 -> buildY ()
    | _ -> (if base < 0 then buildX () else buildY ())
  else
    (let recurse = rand (0, 5) in
     match recurse with
     | 0 -> buildSine (build (rand, (depth - 1)))
     | 1 -> buildCosine (build (rand, (depth - 1)))
     | 2 ->
         buildAverage
           ((build (rand, (depth - 1))), (build (rand, (depth - 1))))
     | 3 ->
         buildTimes
           ((build (rand, (depth - 1))), (build (rand, (depth - 1))))
     | 4 ->
         buildThresh
           ((build (rand, (depth - 1))), (build (rand, (depth - 1))),
             (build (rand, (depth - 1))), (build (rand, (depth - 1))))
     | _ ->
         if recurse > 2
         then buildCosine (build (rand, (depth - 1)))
         else buildSine (build (rand, (depth - 1))));;

*)

(* changed spans
(50,26)-(50,45)
(51,24)-(51,43)
*)

(* type error slice
(13,3)-(13,30)
(13,16)-(13,28)
(13,20)-(13,28)
(13,27)-(13,28)
(15,3)-(15,26)
(15,14)-(15,24)
(15,18)-(15,24)
(15,23)-(15,24)
(50,14)-(50,25)
(50,14)-(50,45)
(50,26)-(50,45)
(51,14)-(51,23)
(51,14)-(51,43)
(51,24)-(51,43)
*)

(* all spans
(11,18)-(11,43)
(11,27)-(11,43)
(11,36)-(11,38)
(11,40)-(11,42)
(13,16)-(13,28)
(13,20)-(13,28)
(13,27)-(13,28)
(15,14)-(15,24)
(15,18)-(15,24)
(15,23)-(15,24)
(17,17)-(17,67)
(17,38)-(17,67)
(17,46)-(17,47)
(17,49)-(17,50)
(17,52)-(17,58)
(17,60)-(17,66)
(19,16)-(19,39)
(19,25)-(19,39)
(19,32)-(19,34)
(19,36)-(19,38)
(21,11)-(21,20)
(21,16)-(21,20)
(23,11)-(23,20)
(23,16)-(23,20)
(25,15)-(51,44)
(26,2)-(51,44)
(26,5)-(26,14)
(26,5)-(26,10)
(26,13)-(26,14)
(28,4)-(32,54)
(28,15)-(28,26)
(28,15)-(28,19)
(28,20)-(28,26)
(28,21)-(28,22)
(28,24)-(28,25)
(29,4)-(32,54)
(29,10)-(29,14)
(30,11)-(30,20)
(30,11)-(30,17)
(30,18)-(30,20)
(31,11)-(31,20)
(31,11)-(31,17)
(31,18)-(31,20)
(32,11)-(32,54)
(32,15)-(32,23)
(32,15)-(32,19)
(32,22)-(32,23)
(32,29)-(32,38)
(32,29)-(32,35)
(32,36)-(32,38)
(32,44)-(32,53)
(32,44)-(32,50)
(32,51)-(32,53)
(34,4)-(51,44)
(34,19)-(34,30)
(34,19)-(34,23)
(34,24)-(34,30)
(34,25)-(34,26)
(34,28)-(34,29)
(35,5)-(51,43)
(35,11)-(35,18)
(36,12)-(36,49)
(36,12)-(36,21)
(36,22)-(36,49)
(36,23)-(36,28)
(36,29)-(36,48)
(36,30)-(36,34)
(36,36)-(36,47)
(36,37)-(36,42)
(36,45)-(36,46)
(37,12)-(37,51)
(37,12)-(37,23)
(37,24)-(37,51)
(37,25)-(37,30)
(37,31)-(37,50)
(37,32)-(37,36)
(37,38)-(37,49)
(37,39)-(37,44)
(37,47)-(37,48)
(39,9)-(40,69)
(39,9)-(39,21)
(40,11)-(40,69)
(40,12)-(40,39)
(40,13)-(40,18)
(40,19)-(40,38)
(40,20)-(40,24)
(40,26)-(40,37)
(40,27)-(40,32)
(40,35)-(40,36)
(40,41)-(40,68)
(40,42)-(40,47)
(40,48)-(40,67)
(40,49)-(40,53)
(40,55)-(40,66)
(40,56)-(40,61)
(40,64)-(40,65)
(42,9)-(43,69)
(42,9)-(42,19)
(43,11)-(43,69)
(43,12)-(43,39)
(43,13)-(43,18)
(43,19)-(43,38)
(43,20)-(43,24)
(43,26)-(43,37)
(43,27)-(43,32)
(43,35)-(43,36)
(43,41)-(43,68)
(43,42)-(43,47)
(43,48)-(43,67)
(43,49)-(43,53)
(43,55)-(43,66)
(43,56)-(43,61)
(43,64)-(43,65)
(45,9)-(47,70)
(45,9)-(45,20)
(46,11)-(47,70)
(46,12)-(46,39)
(46,13)-(46,18)
(46,19)-(46,38)
(46,20)-(46,24)
(46,26)-(46,37)
(46,27)-(46,32)
(46,35)-(46,36)
(46,41)-(46,68)
(46,42)-(46,47)
(46,48)-(46,67)
(46,49)-(46,53)
(46,55)-(46,66)
(46,56)-(46,61)
(46,64)-(46,65)
(47,13)-(47,40)
(47,14)-(47,19)
(47,20)-(47,39)
(47,21)-(47,25)
(47,27)-(47,38)
(47,28)-(47,33)
(47,36)-(47,37)
(47,42)-(47,69)
(47,43)-(47,48)
(47,49)-(47,68)
(47,50)-(47,54)
(47,56)-(47,67)
(47,57)-(47,62)
(47,65)-(47,66)
(49,9)-(51,43)
(49,12)-(49,23)
(49,12)-(49,19)
(49,22)-(49,23)
(50,14)-(50,45)
(50,14)-(50,25)
(50,26)-(50,45)
(50,27)-(50,31)
(50,33)-(50,44)
(50,34)-(50,39)
(50,42)-(50,43)
(51,14)-(51,43)
(51,14)-(51,23)
(51,24)-(51,43)
(51,25)-(51,29)
(51,31)-(51,42)
(51,32)-(51,37)
(51,40)-(51,41)
*)
