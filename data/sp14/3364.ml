
let rec mulByDigit i l =
  match List.rev l with
  | [] -> 0
  | h::t -> [mulByDigit i (List.rev (List.map (fun x  -> x * 10) t)); h * i];;


(* fix

let rec mulByDigit i l =
  match List.rev l with
  | [] -> []
  | h::t ->
      (mulByDigit i (List.rev (List.map (fun x  -> x * 10) t))) @ [h * i];;

*)

(* changed spans
(4,10)-(4,11)
(5,12)-(5,76)
(5,13)-(5,23)
(5,70)-(5,75)
*)

(* type error slice
(3,2)-(5,76)
(4,10)-(4,11)
(5,12)-(5,76)
*)

(* all spans
(2,19)-(5,76)
(2,21)-(5,76)
(3,2)-(5,76)
(3,8)-(3,18)
(3,8)-(3,16)
(3,17)-(3,18)
(4,10)-(4,11)
(5,12)-(5,76)
(5,13)-(5,68)
(5,13)-(5,23)
(5,24)-(5,25)
(5,26)-(5,68)
(5,27)-(5,35)
(5,36)-(5,67)
(5,37)-(5,45)
(5,46)-(5,64)
(5,57)-(5,63)
(5,57)-(5,58)
(5,61)-(5,63)
(5,65)-(5,66)
(5,70)-(5,75)
(5,70)-(5,71)
(5,74)-(5,75)
*)