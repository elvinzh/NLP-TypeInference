
type expr =
  | VarX
  | VarY
  | Sine of expr
  | Cosine of expr
  | Average of expr* expr
  | Times of expr* expr
  | Thresh of expr* expr* expr* expr
  | Trip of expr* expr* expr;;

let pi = 4.0 *. (atan 1.0);;

let rec eval (e,x,y) =
  match e with
  | VarX  -> x
  | VarY  -> y
  | Sine t -> sin (pi *. (eval (t, x, y)))
  | Cosine t -> cos (pi *. (eval (t, x, y)))
  | Average (t,s) -> ((eval (t, x, y)) +. (eval (s, x, y))) /. 2.0
  | Times (t,s) -> (eval (t, x, y)) *. (eval (s, x, y))
  | Thresh (t,r,s,q) ->
      if (eval (t, x, y)) < (eval (r, x, y))
      then eval (s, x, y)
      else eval (q, x, y)
  | Trip (t,r,s) ->
      ((eval (t, x, y)) mod 30.0) +. ((eval (r, x, y)) mod (eval (s, x, y)));;


(* fix

type expr =
  | VarX
  | VarY
  | Sine of expr
  | Cosine of expr
  | Average of expr* expr
  | Times of expr* expr
  | Thresh of expr* expr* expr* expr
  | Trip of expr* expr* expr;;

let pi = 4.0 *. (atan 1.0);;

let rec eval (e,x,y) =
  match e with
  | VarX  -> x
  | VarY  -> y
  | Sine t -> sin (pi *. (eval (t, x, y)))
  | Cosine t -> cos (pi *. (eval (t, x, y)))
  | Average (t,s) -> ((eval (t, x, y)) +. (eval (s, x, y))) /. 2.0
  | Times (t,s) -> (eval (t, x, y)) *. (eval (s, x, y))
  | Thresh (t,r,s,q) ->
      if (eval (t, x, y)) < (eval (r, x, y))
      then eval (s, x, y)
      else eval (q, x, y)
  | Trip (t,r,s) ->
      ((eval (t, x, y)) /. 30.0) +. ((eval (r, x, y)) /. (eval (s, x, y)));;

*)

(* changed spans
(27,6)-(27,33)
(27,37)-(27,76)
*)

(* type error slice
(18,18)-(18,42)
(18,25)-(18,41)
(18,26)-(18,30)
(27,6)-(27,33)
(27,6)-(27,76)
(27,7)-(27,23)
(27,8)-(27,12)
(27,28)-(27,32)
(27,37)-(27,76)
(27,38)-(27,54)
(27,39)-(27,43)
(27,59)-(27,75)
(27,60)-(27,64)
*)

(* all spans
(12,9)-(12,26)
(12,9)-(12,12)
(12,16)-(12,26)
(12,17)-(12,21)
(12,22)-(12,25)
(14,14)-(27,76)
(15,2)-(27,76)
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
(20,21)-(20,66)
(20,21)-(20,59)
(20,22)-(20,38)
(20,23)-(20,27)
(20,28)-(20,37)
(20,29)-(20,30)
(20,32)-(20,33)
(20,35)-(20,36)
(20,42)-(20,58)
(20,43)-(20,47)
(20,48)-(20,57)
(20,49)-(20,50)
(20,52)-(20,53)
(20,55)-(20,56)
(20,63)-(20,66)
(21,19)-(21,55)
(21,19)-(21,35)
(21,20)-(21,24)
(21,25)-(21,34)
(21,26)-(21,27)
(21,29)-(21,30)
(21,32)-(21,33)
(21,39)-(21,55)
(21,40)-(21,44)
(21,45)-(21,54)
(21,46)-(21,47)
(21,49)-(21,50)
(21,52)-(21,53)
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
(27,6)-(27,76)
(27,6)-(27,33)
(27,7)-(27,23)
(27,8)-(27,12)
(27,13)-(27,22)
(27,14)-(27,15)
(27,17)-(27,18)
(27,20)-(27,21)
(27,28)-(27,32)
(27,37)-(27,76)
(27,38)-(27,54)
(27,39)-(27,43)
(27,44)-(27,53)
(27,45)-(27,46)
(27,48)-(27,49)
(27,51)-(27,52)
(27,59)-(27,75)
(27,60)-(27,64)
(27,65)-(27,74)
(27,66)-(27,67)
(27,69)-(27,70)
(27,72)-(27,73)
*)
