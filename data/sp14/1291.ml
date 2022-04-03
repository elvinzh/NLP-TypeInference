
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
  if depth = 0
  then (if (rand (0, 2)) < 1 then buildX () else buildY ())
  else
    (let x = rand (0, 5) in
     if x = 0
     then buildSine (build (rand, (depth - 1)))
     else
       if x = 1
       then buildCosine (build (rand, (depth - 1)))
       else
         if x = 2
         then
           buildAverage
             ((build (rand, (depth - 1))), (build (rand, (depth - 1))))
         else
           if x = 3
           then
             buildTimes
               ((build (rand, (depth - 1))), (build (rand, (depth - 1))))
           else
             if x = 4
             then
               buildThresh
                 ((build (rand, (depth - 1))), (build (rand, (depth - 1))),
                   (build (rand, (depth - 1))), (build (rand, (depth - 1)))));;


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
  if depth = 0
  then (if (rand (0, 2)) < 1 then buildX () else buildY ())
  else
    (let x = rand (0, 5) in
     match x with
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
             (build (rand, (depth - 1))), (build (rand, (depth - 1)))));;

*)

(* changed spans
(30,5)-(50,76)
(30,8)-(30,13)
(30,12)-(30,13)
(33,7)-(50,76)
(33,10)-(33,11)
(33,10)-(33,15)
(33,14)-(33,15)
(36,9)-(50,76)
(36,12)-(36,13)
(36,12)-(36,17)
(36,16)-(36,17)
(41,11)-(50,76)
(41,14)-(41,15)
(41,14)-(41,19)
(41,18)-(41,19)
(46,13)-(50,76)
(46,16)-(46,17)
(46,16)-(46,21)
(46,20)-(46,21)
*)

(* type error slice
(17,3)-(17,69)
(17,17)-(17,67)
(17,38)-(17,67)
(19,3)-(19,41)
(19,16)-(19,39)
(19,25)-(19,39)
(23,3)-(23,22)
(23,11)-(23,20)
(23,16)-(23,20)
(26,2)-(50,77)
(27,7)-(27,59)
(27,49)-(27,55)
(27,49)-(27,58)
(29,4)-(50,77)
(30,5)-(50,76)
(33,7)-(50,76)
(36,9)-(50,76)
(41,11)-(50,76)
(43,13)-(43,23)
(43,13)-(44,73)
(46,13)-(50,76)
(48,15)-(48,26)
(48,15)-(50,76)
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
(25,15)-(50,77)
(26,2)-(50,77)
(26,5)-(26,14)
(26,5)-(26,10)
(26,13)-(26,14)
(27,7)-(27,59)
(27,11)-(27,28)
(27,11)-(27,24)
(27,12)-(27,16)
(27,17)-(27,23)
(27,18)-(27,19)
(27,21)-(27,22)
(27,27)-(27,28)
(27,34)-(27,43)
(27,34)-(27,40)
(27,41)-(27,43)
(27,49)-(27,58)
(27,49)-(27,55)
(27,56)-(27,58)
(29,4)-(50,77)
(29,13)-(29,24)
(29,13)-(29,17)
(29,18)-(29,24)
(29,19)-(29,20)
(29,22)-(29,23)
(30,5)-(50,76)
(30,8)-(30,13)
(30,8)-(30,9)
(30,12)-(30,13)
(31,10)-(31,47)
(31,10)-(31,19)
(31,20)-(31,47)
(31,21)-(31,26)
(31,27)-(31,46)
(31,28)-(31,32)
(31,34)-(31,45)
(31,35)-(31,40)
(31,43)-(31,44)
(33,7)-(50,76)
(33,10)-(33,15)
(33,10)-(33,11)
(33,14)-(33,15)
(34,12)-(34,51)
(34,12)-(34,23)
(34,24)-(34,51)
(34,25)-(34,30)
(34,31)-(34,50)
(34,32)-(34,36)
(34,38)-(34,49)
(34,39)-(34,44)
(34,47)-(34,48)
(36,9)-(50,76)
(36,12)-(36,17)
(36,12)-(36,13)
(36,16)-(36,17)
(38,11)-(39,71)
(38,11)-(38,23)
(39,13)-(39,71)
(39,14)-(39,41)
(39,15)-(39,20)
(39,21)-(39,40)
(39,22)-(39,26)
(39,28)-(39,39)
(39,29)-(39,34)
(39,37)-(39,38)
(39,43)-(39,70)
(39,44)-(39,49)
(39,50)-(39,69)
(39,51)-(39,55)
(39,57)-(39,68)
(39,58)-(39,63)
(39,66)-(39,67)
(41,11)-(50,76)
(41,14)-(41,19)
(41,14)-(41,15)
(41,18)-(41,19)
(43,13)-(44,73)
(43,13)-(43,23)
(44,15)-(44,73)
(44,16)-(44,43)
(44,17)-(44,22)
(44,23)-(44,42)
(44,24)-(44,28)
(44,30)-(44,41)
(44,31)-(44,36)
(44,39)-(44,40)
(44,45)-(44,72)
(44,46)-(44,51)
(44,52)-(44,71)
(44,53)-(44,57)
(44,59)-(44,70)
(44,60)-(44,65)
(44,68)-(44,69)
(46,13)-(50,76)
(46,16)-(46,21)
(46,16)-(46,17)
(46,20)-(46,21)
(48,15)-(50,76)
(48,15)-(48,26)
(49,17)-(50,76)
(49,18)-(49,45)
(49,19)-(49,24)
(49,25)-(49,44)
(49,26)-(49,30)
(49,32)-(49,43)
(49,33)-(49,38)
(49,41)-(49,42)
(49,47)-(49,74)
(49,48)-(49,53)
(49,54)-(49,73)
(49,55)-(49,59)
(49,61)-(49,72)
(49,62)-(49,67)
(49,70)-(49,71)
(50,19)-(50,46)
(50,20)-(50,25)
(50,26)-(50,45)
(50,27)-(50,31)
(50,33)-(50,44)
(50,34)-(50,39)
(50,42)-(50,43)
(50,48)-(50,75)
(50,49)-(50,54)
(50,55)-(50,74)
(50,56)-(50,60)
(50,62)-(50,73)
(50,63)-(50,68)
(50,71)-(50,72)
*)
