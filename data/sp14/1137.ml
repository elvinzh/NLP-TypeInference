
type expr =
  | VarX
  | VarY
  | Sine of expr
  | Cosine of expr
  | Average of expr* expr
  | Times of expr* expr
  | Thresh of expr* expr* expr* expr
  | SquareCosine of expr
  | SquareSinCos of expr* expr* expr;;

let pi = 4.0 *. (atan 1.0);;

let rec eval (e,x,y) =
  match e with
  | VarX  -> x
  | VarY  -> y
  | Sine e' -> sin (pi *. (eval (e', x, y)))
  | Cosine e' -> cos (pi *. (eval (e', x, y)))
  | Average (e1,e2) -> ((eval (e1, x, y)) +. (eval (e2, x, y))) /. 2.0
  | Times (e1,e2) -> (eval (e1, x, y)) *. (eval (e2, x, y))
  | Thresh (e1,e2,e3,e4) ->
      if (eval (e1, x, y)) < (eval (e2, x, y))
      then eval (e3, x, y)
      else eval (e4, x, y)
  | SquareCosine e' -> sqrt (abs_float cos (pi *. (eval (e', x, y))))
  | SquareSinCos (e1,e2,e3) ->
      sqrt
        (abs_float sin
           (((cos (pi *. (eval (e1, x, y)))) *.
               (cos (pi *. (eval (e2, x, y)))))
              *. (cos (pi *. (eval (e3, x, y))))));;


(* fix

type expr =
  | VarX
  | VarY
  | Sine of expr
  | Cosine of expr
  | Average of expr* expr
  | Times of expr* expr
  | Thresh of expr* expr* expr* expr
  | SquareCosine of expr
  | SquareSinCos of expr* expr* expr;;

let pi = 4.0 *. (atan 1.0);;

let rec eval (e,x,y) =
  match e with
  | VarX  -> x
  | VarY  -> y
  | Sine e' -> sin (pi *. (eval (e', x, y)))
  | Cosine e' -> cos (pi *. (eval (e', x, y)))
  | Average (e1,e2) -> ((eval (e1, x, y)) +. (eval (e2, x, y))) /. 2.0
  | Times (e1,e2) -> (eval (e1, x, y)) *. (eval (e2, x, y))
  | Thresh (e1,e2,e3,e4) ->
      if (eval (e1, x, y)) < (eval (e2, x, y))
      then eval (e3, x, y)
      else eval (e4, x, y)
  | SquareCosine e' -> sqrt (abs_float (cos (pi *. (eval (e', x, y)))))
  | SquareSinCos (e1,e2,e3) ->
      sqrt
        (abs_float
           (sin
              (((cos (pi *. (eval (e1, x, y)))) *.
                  (cos (pi *. (eval (e2, x, y)))))
                 *. (cos (pi *. (eval (e3, x, y)))))));;

*)

(* changed spans
(27,28)-(27,69)
(27,39)-(27,42)
(30,8)-(33,50)
(30,19)-(30,22)
*)

(* type error slice
(27,28)-(27,69)
(27,29)-(27,38)
(30,8)-(33,50)
(30,9)-(30,18)
*)

(* all spans
(13,9)-(13,26)
(13,9)-(13,12)
(13,16)-(13,26)
(13,17)-(13,21)
(13,22)-(13,25)
(15,14)-(33,50)
(16,2)-(33,50)
(16,8)-(16,9)
(17,13)-(17,14)
(18,13)-(18,14)
(19,15)-(19,44)
(19,15)-(19,18)
(19,19)-(19,44)
(19,20)-(19,22)
(19,26)-(19,43)
(19,27)-(19,31)
(19,32)-(19,42)
(19,33)-(19,35)
(19,37)-(19,38)
(19,40)-(19,41)
(20,17)-(20,46)
(20,17)-(20,20)
(20,21)-(20,46)
(20,22)-(20,24)
(20,28)-(20,45)
(20,29)-(20,33)
(20,34)-(20,44)
(20,35)-(20,37)
(20,39)-(20,40)
(20,42)-(20,43)
(21,23)-(21,70)
(21,23)-(21,63)
(21,24)-(21,41)
(21,25)-(21,29)
(21,30)-(21,40)
(21,31)-(21,33)
(21,35)-(21,36)
(21,38)-(21,39)
(21,45)-(21,62)
(21,46)-(21,50)
(21,51)-(21,61)
(21,52)-(21,54)
(21,56)-(21,57)
(21,59)-(21,60)
(21,67)-(21,70)
(22,21)-(22,59)
(22,21)-(22,38)
(22,22)-(22,26)
(22,27)-(22,37)
(22,28)-(22,30)
(22,32)-(22,33)
(22,35)-(22,36)
(22,42)-(22,59)
(22,43)-(22,47)
(22,48)-(22,58)
(22,49)-(22,51)
(22,53)-(22,54)
(22,56)-(22,57)
(24,6)-(26,26)
(24,9)-(24,46)
(24,9)-(24,26)
(24,10)-(24,14)
(24,15)-(24,25)
(24,16)-(24,18)
(24,20)-(24,21)
(24,23)-(24,24)
(24,29)-(24,46)
(24,30)-(24,34)
(24,35)-(24,45)
(24,36)-(24,38)
(24,40)-(24,41)
(24,43)-(24,44)
(25,11)-(25,26)
(25,11)-(25,15)
(25,16)-(25,26)
(25,17)-(25,19)
(25,21)-(25,22)
(25,24)-(25,25)
(26,11)-(26,26)
(26,11)-(26,15)
(26,16)-(26,26)
(26,17)-(26,19)
(26,21)-(26,22)
(26,24)-(26,25)
(27,23)-(27,69)
(27,23)-(27,27)
(27,28)-(27,69)
(27,29)-(27,38)
(27,39)-(27,42)
(27,43)-(27,68)
(27,44)-(27,46)
(27,50)-(27,67)
(27,51)-(27,55)
(27,56)-(27,66)
(27,57)-(27,59)
(27,61)-(27,62)
(27,64)-(27,65)
(29,6)-(33,50)
(29,6)-(29,10)
(30,8)-(33,50)
(30,9)-(30,18)
(30,19)-(30,22)
(31,11)-(33,49)
(31,12)-(32,47)
(31,13)-(31,44)
(31,14)-(31,17)
(31,18)-(31,43)
(31,19)-(31,21)
(31,25)-(31,42)
(31,26)-(31,30)
(31,31)-(31,41)
(31,32)-(31,34)
(31,36)-(31,37)
(31,39)-(31,40)
(32,15)-(32,46)
(32,16)-(32,19)
(32,20)-(32,45)
(32,21)-(32,23)
(32,27)-(32,44)
(32,28)-(32,32)
(32,33)-(32,43)
(32,34)-(32,36)
(32,38)-(32,39)
(32,41)-(32,42)
(33,17)-(33,48)
(33,18)-(33,21)
(33,22)-(33,47)
(33,23)-(33,25)
(33,29)-(33,46)
(33,30)-(33,34)
(33,35)-(33,45)
(33,36)-(33,38)
(33,40)-(33,41)
(33,43)-(33,44)
*)
