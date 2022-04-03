
type expr =
  | VarX
  | VarY
  | Sine of expr
  | Cosine of expr
  | Average of expr* expr
  | Times of expr* expr
  | Thresh of expr* expr* expr* expr
  | Op1 of expr
  | Op2 of expr* expr* expr;;

let pi = 4.0 *. (atan 1.0);;

let rec eval (e,x,y) =
  match e with
  | VarX  -> x
  | VarY  -> y
  | Sine e -> sin (pi *. (eval (e, x, y)))
  | Cosine e -> cos (pi *. (eval (e, x, y)))
  | Average (e1,e2) -> ((eval (e1, x, y)) +. (eval (e2, x, y))) /. 2.0
  | Times (e1,e2) -> (eval (e1, x, y)) *. (eval (e2, x, y))
  | Thresh (e1,e2,e3,e4) ->
      if (eval (e1, x, y)) < (eval (e2, x, y))
      then eval (e3, x, y)
      else eval (e4, x, y)
  | Op1 e ->
      (tan (pi *. (eval (e, x, y)))) -.
        ((tan (pi *. (eval (e, x, y)))) / 2.0)
  | Op2 (e1,e2,e3,e4) ->
      if (eval (e1, x, y)) > (eval (e2, x, y))
      then eval (e3, x, y)
      else (eval (e1, x, y)) -. (eval (e2, x, y));;


(* fix

type expr =
  | VarX
  | VarY
  | Sine of expr
  | Cosine of expr
  | Average of expr* expr
  | Times of expr* expr
  | Thresh of expr* expr* expr* expr
  | Op1 of expr
  | Op2 of expr* expr* expr;;

let pi = 4.0 *. (atan 1.0);;

let rec eval (e,x,y) =
  match e with
  | VarX  -> x
  | VarY  -> y
  | Sine e -> sin (pi *. (eval (e, x, y)))
  | Cosine e -> cos (pi *. (eval (e, x, y)))
  | Average (e1,e2) -> ((eval (e1, x, y)) +. (eval (e2, x, y))) /. 2.0
  | Times (e1,e2) -> (eval (e1, x, y)) *. (eval (e2, x, y))
  | Thresh (e1,e2,e3,e4) ->
      if (eval (e1, x, y)) < (eval (e2, x, y))
      then eval (e3, x, y)
      else eval (e4, x, y)
  | Op1 e ->
      (tan (pi *. (eval (e, x, y)))) -.
        ((tan (pi *. (eval (e, x, y)))) /. 2.0)
  | Op2 (e1,e2,e3) ->
      if (eval (e1, x, y)) > (eval (e2, x, y))
      then eval (e3, x, y)
      else (eval (e1, x, y)) -. (eval (e2, x, y));;

*)

(* changed spans
(16,2)-(33,49)
(29,8)-(29,46)
*)

(* type error slice
(28,6)-(29,46)
(29,8)-(29,46)
(29,9)-(29,39)
(29,10)-(29,13)
(29,42)-(29,45)
*)

(* all spans
(13,9)-(13,26)
(13,9)-(13,12)
(13,16)-(13,26)
(13,17)-(13,21)
(13,22)-(13,25)
(15,14)-(33,49)
(16,2)-(33,49)
(16,8)-(16,9)
(17,13)-(17,14)
(18,13)-(18,14)
(19,14)-(19,42)
(19,14)-(19,17)
(19,18)-(19,42)
(19,19)-(19,21)
(19,25)-(19,41)
(19,26)-(19,30)
(19,31)-(19,40)
(19,32)-(19,33)
(19,35)-(19,36)
(19,38)-(19,39)
(20,16)-(20,44)
(20,16)-(20,19)
(20,20)-(20,44)
(20,21)-(20,23)
(20,27)-(20,43)
(20,28)-(20,32)
(20,33)-(20,42)
(20,34)-(20,35)
(20,37)-(20,38)
(20,40)-(20,41)
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
(28,6)-(29,46)
(28,6)-(28,36)
(28,7)-(28,10)
(28,11)-(28,35)
(28,12)-(28,14)
(28,18)-(28,34)
(28,19)-(28,23)
(28,24)-(28,33)
(28,25)-(28,26)
(28,28)-(28,29)
(28,31)-(28,32)
(29,8)-(29,46)
(29,9)-(29,39)
(29,10)-(29,13)
(29,14)-(29,38)
(29,15)-(29,17)
(29,21)-(29,37)
(29,22)-(29,26)
(29,27)-(29,36)
(29,28)-(29,29)
(29,31)-(29,32)
(29,34)-(29,35)
(29,42)-(29,45)
(31,6)-(33,49)
(31,9)-(31,46)
(31,9)-(31,26)
(31,10)-(31,14)
(31,15)-(31,25)
(31,16)-(31,18)
(31,20)-(31,21)
(31,23)-(31,24)
(31,29)-(31,46)
(31,30)-(31,34)
(31,35)-(31,45)
(31,36)-(31,38)
(31,40)-(31,41)
(31,43)-(31,44)
(32,11)-(32,26)
(32,11)-(32,15)
(32,16)-(32,26)
(32,17)-(32,19)
(32,21)-(32,22)
(32,24)-(32,25)
(33,11)-(33,49)
(33,11)-(33,28)
(33,12)-(33,16)
(33,17)-(33,27)
(33,18)-(33,20)
(33,22)-(33,23)
(33,25)-(33,26)
(33,32)-(33,49)
(33,33)-(33,37)
(33,38)-(33,48)
(33,39)-(33,41)
(33,43)-(33,44)
(33,46)-(33,47)
*)
