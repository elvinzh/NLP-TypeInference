
let rec mulByDigit i l =
  match List.rev l with
  | [] -> []
  | h::t ->
      (match (mulByDigit i (List.rev (List.map (fun x  -> x * 10) t))) @
               [h * i]
       with
       | [] -> []
       | h::t ->
           let f a x = a + x in
           let base = 0 in List.fold_left f base (h :: t));;


(* fix

let rec mulByDigit i l =
  match List.rev l with
  | [] -> []
  | h::t ->
      (match (mulByDigit i (List.rev (List.map (fun x  -> x * 10) t))) @
               [h * i]
       with
       | [] -> []
       | h::t ->
           let f a x = a + x in
           let base = 0 in [List.fold_left f base (h :: t)]);;

*)

(* changed spans
(12,27)-(12,57)
*)

(* type error slice
(6,6)-(12,58)
(9,15)-(9,17)
(11,11)-(12,57)
(11,17)-(11,28)
(11,23)-(11,24)
(11,23)-(11,28)
(12,11)-(12,57)
(12,27)-(12,41)
(12,27)-(12,57)
(12,42)-(12,43)
*)

(* all spans
(2,19)-(12,58)
(2,21)-(12,58)
(3,2)-(12,58)
(3,8)-(3,18)
(3,8)-(3,16)
(3,17)-(3,18)
(4,10)-(4,12)
(6,6)-(12,58)
(6,13)-(7,22)
(6,71)-(6,72)
(6,13)-(6,70)
(6,14)-(6,24)
(6,25)-(6,26)
(6,27)-(6,69)
(6,28)-(6,36)
(6,37)-(6,68)
(6,38)-(6,46)
(6,47)-(6,65)
(6,58)-(6,64)
(6,58)-(6,59)
(6,62)-(6,64)
(6,66)-(6,67)
(7,15)-(7,22)
(7,16)-(7,21)
(7,16)-(7,17)
(7,20)-(7,21)
(9,15)-(9,17)
(11,11)-(12,57)
(11,17)-(11,28)
(11,19)-(11,28)
(11,23)-(11,28)
(11,23)-(11,24)
(11,27)-(11,28)
(12,11)-(12,57)
(12,22)-(12,23)
(12,27)-(12,57)
(12,27)-(12,41)
(12,42)-(12,43)
(12,44)-(12,48)
(12,49)-(12,57)
(12,50)-(12,51)
(12,55)-(12,56)
*)
