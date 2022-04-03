
type expr =
  | VarX
  | VarY
  | Sine of expr
  | Cosine of expr
  | Average of expr* expr
  | Times of expr* expr
  | Thresh of expr* expr* expr* expr
  | Power of expr* expr;;

let pi = 4.0 *. (atan 1.0);;

let rec eval (e,x,y) =
  match e with
  | VarX  -> x
  | VarY  -> y
  | Sine a -> sin (pi *. (eval (a, x, y)))
  | Cosine a -> cos (pi *. (eval (a, x, y)))
  | Average (e1,e2) -> ((eval (e1, x, y)) +. (eval (e2, x, y))) /. 2.0
  | Times (e1,e2) -> (eval (e1, x, y)) *. (eval (e2, x, y))
  | Thresh (a,b,c,d) ->
      if (eval (a, x, y)) < (eval (b, x, y))
      then eval (c, x, y)
      else eval (d, x, y)
  | Power (a,b) ->
      if
        (((-1.0) < x) < 1.0) && ((x > (-1.0)) && ((y < 1.0) && (y > (-1.0))))
      then x *. y
      else (eval (a, x, y)) ** (eval (b, x, y));;


(* fix

type expr =
  | VarX
  | VarY
  | Sine of expr
  | Cosine of expr
  | Average of expr* expr
  | Times of expr* expr
  | Thresh of expr* expr* expr* expr
  | Power of expr* expr;;

let pi = 4.0 *. (atan 1.0);;

let rec eval (e,x,y) =
  match e with
  | VarX  -> x
  | VarY  -> y
  | Sine a -> sin (pi *. (eval (a, x, y)))
  | Cosine a -> cos (pi *. (eval (a, x, y)))
  | Average (e1,e2) -> ((eval (e1, x, y)) +. (eval (e2, x, y))) /. 2.0
  | Times (e1,e2) -> (eval (e1, x, y)) *. (eval (e2, x, y))
  | Thresh (a,b,c,d) ->
      if (eval (a, x, y)) < (eval (b, x, y))
      then eval (c, x, y)
      else eval (d, x, y)
  | Power (a,b) -> (eval (a, x, y)) +. (eval (b, x, y));;

*)

(* changed spans
(27,6)-(30,47)
(28,8)-(28,28)
(28,8)-(28,77)
(28,9)-(28,21)
(28,10)-(28,16)
(28,12)-(28,15)
(28,19)-(28,20)
(28,24)-(28,27)
(28,32)-(28,77)
(28,33)-(28,45)
(28,34)-(28,35)
(28,38)-(28,44)
(28,40)-(28,43)
(28,49)-(28,76)
(28,50)-(28,59)
(28,51)-(28,52)
(28,55)-(28,58)
(28,63)-(28,75)
(28,64)-(28,65)
(28,68)-(28,74)
(28,70)-(28,73)
(29,11)-(29,12)
(29,11)-(29,17)
(29,16)-(29,17)
(30,11)-(30,47)
(30,28)-(30,30)
*)

(* type error slice
(14,3)-(30,49)
(14,14)-(30,47)
(15,2)-(30,47)
(16,13)-(16,14)
(17,13)-(17,14)
(18,14)-(18,17)
(18,14)-(18,42)
(18,18)-(18,42)
(18,25)-(18,41)
(18,26)-(18,30)
(19,16)-(19,19)
(19,16)-(19,44)
(20,23)-(20,70)
(21,21)-(21,59)
(23,6)-(25,25)
(25,11)-(25,15)
(25,11)-(25,25)
(27,6)-(30,47)
(28,8)-(28,28)
(28,9)-(28,21)
(28,10)-(28,16)
(28,12)-(28,15)
(28,19)-(28,20)
(28,24)-(28,27)
(28,38)-(28,44)
(28,40)-(28,43)
(28,50)-(28,59)
(28,51)-(28,52)
(28,55)-(28,58)
(28,63)-(28,75)
(28,64)-(28,65)
(28,68)-(28,74)
(28,70)-(28,73)
(29,11)-(29,12)
(29,11)-(29,17)
(30,11)-(30,47)
(30,28)-(30,30)
*)

(* all spans
(12,9)-(12,26)
(12,9)-(12,12)
(12,16)-(12,26)
(12,17)-(12,21)
(12,22)-(12,25)
(14,14)-(30,47)
(15,2)-(30,47)
(15,8)-(15,9)
(16,13)-(16,14)
(17,13)-(17,14)
(18,14)-(18,42)
(18,14)-(18,17)
(18,18)-(18,42)
(18,19)-(18,21)
(18,25)-(18,41)
(18,26)-(18,30)
(18,31)-(18,40)
(18,32)-(18,33)
(18,35)-(18,36)
(18,38)-(18,39)
(19,16)-(19,44)
(19,16)-(19,19)
(19,20)-(19,44)
(19,21)-(19,23)
(19,27)-(19,43)
(19,28)-(19,32)
(19,33)-(19,42)
(19,34)-(19,35)
(19,37)-(19,38)
(19,40)-(19,41)
(20,23)-(20,70)
(20,23)-(20,63)
(20,24)-(20,41)
(20,25)-(20,29)
(20,30)-(20,40)
(20,31)-(20,33)
(20,35)-(20,36)
(20,38)-(20,39)
(20,45)-(20,62)
(20,46)-(20,50)
(20,51)-(20,61)
(20,52)-(20,54)
(20,56)-(20,57)
(20,59)-(20,60)
(20,67)-(20,70)
(21,21)-(21,59)
(21,21)-(21,38)
(21,22)-(21,26)
(21,27)-(21,37)
(21,28)-(21,30)
(21,32)-(21,33)
(21,35)-(21,36)
(21,42)-(21,59)
(21,43)-(21,47)
(21,48)-(21,58)
(21,49)-(21,51)
(21,53)-(21,54)
(21,56)-(21,57)
(23,6)-(25,25)
(23,9)-(23,44)
(23,9)-(23,25)
(23,10)-(23,14)
(23,15)-(23,24)
(23,16)-(23,17)
(23,19)-(23,20)
(23,22)-(23,23)
(23,28)-(23,44)
(23,29)-(23,33)
(23,34)-(23,43)
(23,35)-(23,36)
(23,38)-(23,39)
(23,41)-(23,42)
(24,11)-(24,25)
(24,11)-(24,15)
(24,16)-(24,25)
(24,17)-(24,18)
(24,20)-(24,21)
(24,23)-(24,24)
(25,11)-(25,25)
(25,11)-(25,15)
(25,16)-(25,25)
(25,17)-(25,18)
(25,20)-(25,21)
(25,23)-(25,24)
(27,6)-(30,47)
(28,8)-(28,77)
(28,8)-(28,28)
(28,9)-(28,21)
(28,10)-(28,16)
(28,12)-(28,15)
(28,19)-(28,20)
(28,24)-(28,27)
(28,32)-(28,77)
(28,33)-(28,45)
(28,34)-(28,35)
(28,38)-(28,44)
(28,40)-(28,43)
(28,49)-(28,76)
(28,50)-(28,59)
(28,51)-(28,52)
(28,55)-(28,58)
(28,63)-(28,75)
(28,64)-(28,65)
(28,68)-(28,74)
(28,70)-(28,73)
(29,11)-(29,17)
(29,11)-(29,12)
(29,16)-(29,17)
(30,11)-(30,47)
(30,28)-(30,30)
(30,11)-(30,27)
(30,12)-(30,16)
(30,17)-(30,26)
(30,18)-(30,19)
(30,21)-(30,22)
(30,24)-(30,25)
(30,31)-(30,47)
(30,32)-(30,36)
(30,37)-(30,46)
(30,38)-(30,39)
(30,41)-(30,42)
(30,44)-(30,45)
*)
