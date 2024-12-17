import Link from "next/link";
import Image from "next/image";
import { MasonryProps } from "./Interface";

export default function Masonry({ cards }: MasonryProps) {
  return (
    <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 max-w-6xl mx-auto p-4">
      {cards.map((card, index) => (
        <div
          key={index}
          className="relative overflow-hidden rounded-lg shadow-lg group hover:shadow-xl hover:-translate-y-2 transition-transform duration-300 ease-in-out"
        >
          <Link href="#" className="absolute inset-0 z-10" prefetch={true}>
            <span className="sr-only">View</span>
          </Link>
          <Image
            src={card.src}
            alt={card.title}
            width={card.width}
            height={card.height}
            className={`object-cover w-full ${card.heightClass}`}
            style={{
              aspectRatio: `${card.width}/${card.height}`,
              objectFit: "cover",
            }}
          />
          <div className="p-4 bg-background">
            <h3 className="text-xl font-bold">{card.title}</h3>
            <p className="text-sm text-muted-foreground">{card.description}</p>
          </div>
        </div>
      ))}
    </div>
  );
}
