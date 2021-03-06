
type expr =
  | VarX
  | VarY
  | Sine of expr
  | Cosine of expr
  | Average of expr* expr
  | Times of expr* expr
  | Thresh of expr* expr* expr* expr
  | Abs of expr
  | Flip of expr* expr* expr;;

let pi = 4.0 *. (atan 1.0);;

let rec eval (e,x,y) =
  match e with
  | VarX  -> x
  | VarY  -> y
  | Sine q -> sin (pi *. (eval (q, x, y)))
  | Cosine q -> cos (pi *. (eval (q, x, y)))
  | Average (e1,e2) -> ((eval (e1, x, y)) +. (eval (e2, x, y))) /. 2.0
  | Times (e1,e2) -> (eval (e1, x, y)) *. (eval (e2, x, y))
  | Thresh (a,b,a_less,b_less) ->
      if (eval (a, x, y)) < (eval (b, x, y))
      then eval (a_less, x, y)
      else eval (b_less, x, y)
  | Abs v ->
      if (eval (v, x, y)) < 0
      then (eval (v, x, y)) *. (-1)
      else eval (v, x, y)
  | Flip (a,b,c) ->
      if (eval (a, x, y)) > (eval (b, x, y))
      then eval ((c *. (-1)), x, y)
      else eval (c, x, y);;


(* fix

type expr =
  | VarX
  | VarY
  | Sine of expr
  | Cosine of expr
  | Average of expr* expr
  | Times of expr* expr
  | Thresh of expr* expr* expr* expr
  | Mid of expr* expr
  | Flip of expr* expr* expr;;

let pi = 4.0 *. (atan 1.0);;

let rec eval (e,x,y) =
  match e with
  | VarX  -> x
  | VarY  -> y
  | Sine q -> sin (pi *. (eval (q, x, y)))
  | Cosine q -> cos (pi *. (eval (q, x, y)))
  | Average (e1,e2) -> ((eval (e1, x, y)) +. (eval (e2, x, y))) /. 2.0
  | Times (e1,e2) -> (eval (e1, x, y)) *. (eval (e2, x, y))
  | Thresh (a,b,a_less,b_less) ->
      if (eval (a, x, y)) < (eval (b, x, y))
      then eval (a_less, x, y)
      else eval (b_less, x, y)
  | Mid (p,q) ->
      let diff =
        if ((eval (p, x, y)) -. (eval (q, x, y))) < 0.0
        then (eval (p, x, y)) -. ((eval (q, x, y)) *. (-1.0))
        else (eval (p, x, y)) -. (eval (q, x, y)) in
      diff /. 2.0
  | Flip (a,b,c) ->
      if (eval (a, x, y)) > (eval (b, x, y))
      then (eval (c, x, y)) *. (-1.0)
      else eval (c, x, y);;

*)

(* changed spans
(16,2)-(34,25)
(28,6)-(30,25)
(28,9)-(28,25)
(28,16)-(28,17)
(28,28)-(28,29)
(29,11)-(29,35)
(29,18)-(29,19)
(29,31)-(29,35)
(30,11)-(30,25)
(30,17)-(30,18)
(32,6)-(34,25)
(33,11)-(33,35)
(33,17)-(33,28)
(33,23)-(33,27)
(34,11)-(34,25)
*)

(* type error slice
(16,2)-(34,25)
(19,18)-(19,42)
(19,25)-(19,41)
(19,26)-(19,30)
(19,31)-(19,40)
(19,32)-(19,33)
(28,9)-(28,25)
(28,9)-(28,29)
(28,10)-(28,14)
(28,28)-(28,29)
(29,11)-(29,35)
(29,31)-(29,35)
(33,11)-(33,15)
(33,11)-(33,35)
(33,16)-(33,35)
(33,17)-(33,28)
(33,18)-(33,19)
(33,23)-(33,27)
*)

(* all spans
(13,9)-(13,26)
(13,9)-(13,12)
(13,16)-(13,26)
(13,17)-(13,21)
(13,22)-(13,25)
(15,14)-(34,25)
(16,2)-(34,25)
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
(24,6)-(26,30)
(24,9)-(24,44)
(24,9)-(24,25)
(24,10)-(24,14)
(24,15)-(24,24)
(24,16)-(24,17)
(24,19)-(24,20)
(24,22)-(24,23)
(24,28)-(24,44)
(24,29)-(24,33)
(24,34)-(24,43)
(24,35)-(24,36)
(24,38)-(24,39)
(24,41)-(24,42)
(25,11)-(25,30)
(25,11)-(25,15)
(25,16)-(25,30)
(25,17)-(25,23)
(25,25)-(25,26)
(25,28)-(25,29)
(26,11)-(26,30)
(26,11)-(26,15)
(26,16)-(26,30)
(26,17)-(26,23)
(26,25)-(26,26)
(26,28)-(26,29)
(28,6)-(30,25)
(28,9)-(28,29)
(28,9)-(28,25)
(28,10)-(28,14)
(28,15)-(28,24)
(28,16)-(28,17)
(28,19)-(28,20)
(28,22)-(28,23)
(28,28)-(28,29)
(29,11)-(29,35)
(29,11)-(29,27)
(29,12)-(29,16)
(29,17)-(29,26)
(29,18)-(29,19)
(29,21)-(29,22)
(29,24)-(29,25)
(29,31)-(29,35)
(30,11)-(30,25)
(30,11)-(30,15)
(30,16)-(30,25)
(30,17)-(30,18)
(30,20)-(30,21)
(30,23)-(30,24)
(32,6)-(34,25)
(32,9)-(32,44)
(32,9)-(32,25)
(32,10)-(32,14)
(32,15)-(32,24)
(32,16)-(32,17)
(32,19)-(32,20)
(32,22)-(32,23)
(32,28)-(32,44)
(32,29)-(32,33)
(32,34)-(32,43)
(32,35)-(32,36)
(32,38)-(32,39)
(32,41)-(32,42)
(33,11)-(33,35)
(33,11)-(33,15)
(33,16)-(33,35)
(33,17)-(33,28)
(33,18)-(33,19)
(33,23)-(33,27)
(33,30)-(33,31)
(33,33)-(33,34)
(34,11)-(34,25)
(34,11)-(34,15)
(34,16)-(34,25)
(34,17)-(34,18)
(34,20)-(34,21)
(34,23)-(34,24)
*)
