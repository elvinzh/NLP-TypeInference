
type expr =
  | VarX
  | VarY
  | Sine of expr
  | Cosine of expr
  | Average of expr* expr
  | Times of expr* expr
  | Thresh of expr* expr* expr* expr;;

let pi = 4.0 *. (atan 1.0);;

let rec eval (e,x,y) =
  match e with
  | VarX  -> x
  | VarY  -> y
  | Sine expr -> sin (pi *. (eval (expr, x, y)))
  | Cosine expr -> cos (pi *. (eval (expr, x, y)))
  | Average (expr,expr1) -> ((eval (expr, x, y)) +. (eval (expr1, x, y))) / 2
  | Times (expr,expr1) -> (eval (expr, x, y)) *. (eval (expr1, x, y))
  | Thresh (expr,expr1,expr2,expr3) ->
      if (eval (expr, x, y)) < (eval (expr1, x, y))
      then eval (expr2, x, y)
      else eval (expr3, x, y);;


(* fix

type expr =
  | VarX
  | VarY
  | Sine of expr
  | Cosine of expr
  | Average of expr* expr
  | Times of expr* expr
  | Thresh of expr* expr* expr* expr;;

let pi = 4.0 *. (atan 1.0);;

let rec eval (e,x,y) =
  match e with
  | VarX  -> x
  | VarY  -> y
  | Sine expr -> sin (pi *. (eval (expr, x, y)))
  | Cosine expr -> cos (pi *. (eval (expr, x, y)))
  | Average (expr,expr1) ->
      ((eval (expr, x, y)) +. (eval (expr1, x, y))) /. 2.
  | Times (expr,expr1) -> (eval (expr, x, y)) *. (eval (expr1, x, y))
  | Thresh (expr,expr1,expr2,expr3) ->
      if (eval (expr, x, y)) < (eval (expr1, x, y))
      then eval (expr2, x, y)
      else eval (expr3, x, y);;

*)

(* changed spans
(19,28)-(19,77)
(19,76)-(19,77)
*)

(* type error slice
(14,2)-(24,29)
(17,17)-(17,20)
(17,17)-(17,48)
(19,28)-(19,73)
(19,28)-(19,77)
*)

(* all spans
(11,9)-(11,26)
(11,9)-(11,12)
(11,16)-(11,26)
(11,17)-(11,21)
(11,22)-(11,25)
(13,14)-(24,29)
(14,2)-(24,29)
(14,8)-(14,9)
(15,13)-(15,14)
(16,13)-(16,14)
(17,17)-(17,48)
(17,17)-(17,20)
(17,21)-(17,48)
(17,22)-(17,24)
(17,28)-(17,47)
(17,29)-(17,33)
(17,34)-(17,46)
(17,35)-(17,39)
(17,41)-(17,42)
(17,44)-(17,45)
(18,19)-(18,50)
(18,19)-(18,22)
(18,23)-(18,50)
(18,24)-(18,26)
(18,30)-(18,49)
(18,31)-(18,35)
(18,36)-(18,48)
(18,37)-(18,41)
(18,43)-(18,44)
(18,46)-(18,47)
(19,28)-(19,77)
(19,28)-(19,73)
(19,29)-(19,48)
(19,30)-(19,34)
(19,35)-(19,47)
(19,36)-(19,40)
(19,42)-(19,43)
(19,45)-(19,46)
(19,52)-(19,72)
(19,53)-(19,57)
(19,58)-(19,71)
(19,59)-(19,64)
(19,66)-(19,67)
(19,69)-(19,70)
(19,76)-(19,77)
(20,26)-(20,69)
(20,26)-(20,45)
(20,27)-(20,31)
(20,32)-(20,44)
(20,33)-(20,37)
(20,39)-(20,40)
(20,42)-(20,43)
(20,49)-(20,69)
(20,50)-(20,54)
(20,55)-(20,68)
(20,56)-(20,61)
(20,63)-(20,64)
(20,66)-(20,67)
(22,6)-(24,29)
(22,9)-(22,51)
(22,9)-(22,28)
(22,10)-(22,14)
(22,15)-(22,27)
(22,16)-(22,20)
(22,22)-(22,23)
(22,25)-(22,26)
(22,31)-(22,51)
(22,32)-(22,36)
(22,37)-(22,50)
(22,38)-(22,43)
(22,45)-(22,46)
(22,48)-(22,49)
(23,11)-(23,29)
(23,11)-(23,15)
(23,16)-(23,29)
(23,17)-(23,22)
(23,24)-(23,25)
(23,27)-(23,28)
(24,11)-(24,29)
(24,11)-(24,15)
(24,16)-(24,29)
(24,17)-(24,22)
(24,24)-(24,25)
(24,27)-(24,28)
*)
