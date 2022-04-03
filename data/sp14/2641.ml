
type expr =
  | VarX
  | VarY
  | Sine of expr
  | Cosine of expr
  | Average of expr* expr
  | Times of expr* expr
  | Thresh of expr* expr* expr* expr
  | Factorial of expr
  | Sum3 of expr* expr* expr;;

let rec factorial x acc =
  if x = 0.0 then acc else factorial (x -. 1.0) (x *. acc);;

let pi = 4.0 *. (atan 1.0);;

let rec eval (e,x,y) =
  match e with
  | VarX  -> x
  | VarY  -> y
  | Sine e' -> sin (pi *. (eval (e', x, y)))
  | Cosine e' -> cos (pi *. (eval (e', x, y)))
  | Average (e1,e2) -> ((eval (e1, x, y)) +. (eval (e2, x, y))) /. 2.0
  | Times (e1,e2) -> (eval (e1, x, y)) *. (eval (e2, x, y))
  | Thresh (a,b,a_less,b_less) ->
      if (eval (a, x, y)) < (eval (b, x, y))
      then eval (a_less, x, y)
      else eval (b_less, x, y)
  | Factorial e' -> factorial (eval e')
  | Sum3 (e1,e2,e3) -> ((eval e1) +. (eval e2)) +. (eval e3);;


(* fix

type expr =
  | VarX
  | VarY
  | Sine of expr
  | Cosine of expr
  | Average of expr* expr
  | Times of expr* expr
  | Thresh of expr* expr* expr* expr
  | Factorial of expr
  | Sum3 of expr* expr* expr;;

let pi = 4.0 *. (atan 1.0);;

let rec eval (e,x,y) =
  match e with
  | VarX  -> x
  | VarY  -> y
  | Sine e' -> sin (pi *. (eval (e', x, y)))
  | Cosine e' -> cos (pi *. (eval (e', x, y)))
  | Average (e1,e2) -> ((eval (e1, x, y)) +. (eval (e2, x, y))) /. 2.0
  | Times (e1,e2) -> (eval (e1, x, y)) *. (eval (e2, x, y))
  | Thresh (a,b,a_less,b_less) ->
      if (eval (a, x, y)) < (eval (b, x, y))
      then eval (a_less, x, y)
      else eval (b_less, x, y);;

*)

(* changed spans
(13,18)-(14,58)
(13,20)-(14,58)
(14,2)-(14,58)
(14,5)-(14,6)
(14,5)-(14,12)
(14,9)-(14,12)
(14,18)-(14,21)
(14,27)-(14,36)
(14,27)-(14,58)
(14,37)-(14,47)
(14,38)-(14,39)
(14,43)-(14,46)
(14,49)-(14,50)
(14,54)-(14,57)
(16,9)-(16,26)
(19,2)-(31,60)
(30,20)-(30,29)
(30,20)-(30,39)
(30,30)-(30,39)
(30,31)-(30,35)
(30,36)-(30,38)
(31,23)-(31,47)
(31,23)-(31,60)
(31,24)-(31,33)
(31,25)-(31,29)
(31,30)-(31,32)
(31,37)-(31,46)
(31,38)-(31,42)
(31,43)-(31,45)
(31,51)-(31,60)
(31,52)-(31,56)
(31,57)-(31,59)
*)

(* type error slice
(14,27)-(14,36)
(14,27)-(14,58)
(19,2)-(31,60)
(22,15)-(22,18)
(22,15)-(22,44)
(22,26)-(22,43)
(22,27)-(22,31)
(22,32)-(22,42)
(30,20)-(30,29)
(30,20)-(30,39)
(30,30)-(30,39)
(30,31)-(30,35)
(30,36)-(30,38)
(31,24)-(31,33)
(31,25)-(31,29)
(31,30)-(31,32)
(31,37)-(31,46)
(31,38)-(31,42)
(31,43)-(31,45)
(31,51)-(31,60)
(31,52)-(31,56)
(31,57)-(31,59)
*)

(* all spans
(13,18)-(14,58)
(13,20)-(14,58)
(14,2)-(14,58)
(14,5)-(14,12)
(14,5)-(14,6)
(14,9)-(14,12)
(14,18)-(14,21)
(14,27)-(14,58)
(14,27)-(14,36)
(14,37)-(14,47)
(14,38)-(14,39)
(14,43)-(14,46)
(14,48)-(14,58)
(14,49)-(14,50)
(14,54)-(14,57)
(16,9)-(16,26)
(16,9)-(16,12)
(16,16)-(16,26)
(16,17)-(16,21)
(16,22)-(16,25)
(18,14)-(31,60)
(19,2)-(31,60)
(19,8)-(19,9)
(20,13)-(20,14)
(21,13)-(21,14)
(22,15)-(22,44)
(22,15)-(22,18)
(22,19)-(22,44)
(22,20)-(22,22)
(22,26)-(22,43)
(22,27)-(22,31)
(22,32)-(22,42)
(22,33)-(22,35)
(22,37)-(22,38)
(22,40)-(22,41)
(23,17)-(23,46)
(23,17)-(23,20)
(23,21)-(23,46)
(23,22)-(23,24)
(23,28)-(23,45)
(23,29)-(23,33)
(23,34)-(23,44)
(23,35)-(23,37)
(23,39)-(23,40)
(23,42)-(23,43)
(24,23)-(24,70)
(24,23)-(24,63)
(24,24)-(24,41)
(24,25)-(24,29)
(24,30)-(24,40)
(24,31)-(24,33)
(24,35)-(24,36)
(24,38)-(24,39)
(24,45)-(24,62)
(24,46)-(24,50)
(24,51)-(24,61)
(24,52)-(24,54)
(24,56)-(24,57)
(24,59)-(24,60)
(24,67)-(24,70)
(25,21)-(25,59)
(25,21)-(25,38)
(25,22)-(25,26)
(25,27)-(25,37)
(25,28)-(25,30)
(25,32)-(25,33)
(25,35)-(25,36)
(25,42)-(25,59)
(25,43)-(25,47)
(25,48)-(25,58)
(25,49)-(25,51)
(25,53)-(25,54)
(25,56)-(25,57)
(27,6)-(29,30)
(27,9)-(27,44)
(27,9)-(27,25)
(27,10)-(27,14)
(27,15)-(27,24)
(27,16)-(27,17)
(27,19)-(27,20)
(27,22)-(27,23)
(27,28)-(27,44)
(27,29)-(27,33)
(27,34)-(27,43)
(27,35)-(27,36)
(27,38)-(27,39)
(27,41)-(27,42)
(28,11)-(28,30)
(28,11)-(28,15)
(28,16)-(28,30)
(28,17)-(28,23)
(28,25)-(28,26)
(28,28)-(28,29)
(29,11)-(29,30)
(29,11)-(29,15)
(29,16)-(29,30)
(29,17)-(29,23)
(29,25)-(29,26)
(29,28)-(29,29)
(30,20)-(30,39)
(30,20)-(30,29)
(30,30)-(30,39)
(30,31)-(30,35)
(30,36)-(30,38)
(31,23)-(31,60)
(31,23)-(31,47)
(31,24)-(31,33)
(31,25)-(31,29)
(31,30)-(31,32)
(31,37)-(31,46)
(31,38)-(31,42)
(31,43)-(31,45)
(31,51)-(31,60)
(31,52)-(31,56)
(31,57)-(31,59)
*)